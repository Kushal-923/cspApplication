"""
scheduler_1.2.py  —  Two-Phase CSP / ILP Radar Scheduler
=========================================================

Changes over v1.1:
  1. BUDGETS are auto-generated from radar metadata (no hardcoded list).
     Budget rule: Large=18, Medium=16, Small=12 active slots out of M=24.
     Override by editing BUDGET_BY_TYPE below.
  2. NUM_TIME_SLOTS (M) is still editable — default stays 24.
  3. Prints coverage statistics before solving (how many points each radar
     covers, what fraction of the border is reachable at all).
  4. Minor: status summary prints void % and overlap density for easier
     comparison across runs.

Algorithm is unchanged — see scheduler_1.1.py for full documentation.

Input files:
  c_matrix.json     — built by build_matrix_1.2.py
  radar_meta.json   — built by build_matrix_1.2.py
"""

import json
from ortools.sat.python import cp_model

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

MATRIX_FILE = "c_matrix.json"
META_FILE   = "radar_meta.json"

NUM_TIME_SLOTS = 24   # M — 24 hourly slots = one full day

# Cooling constraint
#   In any rolling window of (L + C_COOL) slots,
#   a radar may be ON for at most L slots.
L      = 3
C_COOL = 2    # window = 5: at most 3 ON per 5 consecutive slots

# Energy budget per radar type (max total ON-slots over all M slots).
# Keyed on the substring that appears in the radar name.
# Edit these to match your hardware thermal specs.
BUDGET_BY_TYPE = {
    "Large":  18,   # can run 18 out of 24 hours
    "Medium": 16,
    "Small":  12,
}
DEFAULT_BUDGET = 14   # fallback if name matches none of the above

TIME_LIMIT = 300.0    # seconds per solver phase — increase for large K


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def banner(title, subtitle=""):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(f"{'─'*60}")


def status_label(code):
    return {
        cp_model.OPTIMAL:    "OPTIMAL",
        cp_model.FEASIBLE:   "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN:    "UNKNOWN (time-out?)",
    }.get(code, "UNKNOWN")


def assign_budgets(meta, M):
    """
    Derive per-radar energy budget from radar name.
    Also enforces budget <= M (can't be on more than M slots).
    """
    budgets = []
    for r in meta:
        b = DEFAULT_BUDGET
        for key, val in BUDGET_BY_TYPE.items():
            if key in r["name"]:
                b = val
                break
        budgets.append(min(b, M))
    return budgets


def add_cooling_and_energy(model, X, N, M, L, C_cool, B):
    """
    Cooling  : Σ_{k=t}^{min(t+L+C-1, M-1)} X[i,k]  ≤  L   ∀ i, t
    Energy   : Σ_t X[i,t]                            ≤  B[i] ∀ i
    """
    window = L + C_cool
    if window > 1:
        for i in range(N):
            for t in range(M):
                end = min(t + window, M)
                model.Add(sum(X[i][k] for k in range(t, end)) <= L)
    for i in range(N):
        model.Add(sum(X[i][t] for t in range(M)) <= B[i])


def print_schedule(sched, N, M, B, meta):
    """Print ON/OFF schedule. Full grid for M<=24, summary bars for larger M."""
    print("\n  FINAL SCHEDULE")
    if M <= 24:
        header = "                    " + " ".join(f"{t:2d}" for t in range(M))
        print(header)
        for i in range(N):
            on  = sum(sched[i])
            name = meta[i]["name"] if meta else f"R{i+1:02d}"
            row  = "  ".join(str(sched[i][t]) for t in range(M))
            print(f"  {name:14s} [{on:2d}/{B[i]}]  {row}")
    else:
        print("  (M > 24 — showing summary bars)")
        for i in range(N):
            on  = sum(sched[i])
            pct = 100 * on / M
            bar = "█" * int(pct / 5)
            name = meta[i]["name"] if meta else f"R{i+1:02d}"
            print(f"  {name:14s} [{on:3d}/{B[i]}]  ({pct:4.1f}%)  {bar}")


def print_coverage_stats(C_mat, N, K):
    """Show how many boundary points each radar covers and overall reachability."""
    banner("COVERAGE MATRIX STATISTICS")
    for i in range(N):
        covered = sum(C_mat[i])
        pct     = 100 * covered / K
        bar     = "█" * int(pct / 5)
        print(f"  R{i+1:02d}: {covered:4d}/{K} pts  ({pct:5.1f}%)  {bar}")

    # Column sums: how many radars cover each point
    col_sums = [sum(C_mat[i][j] for i in range(N)) for j in range(K)]
    uncovered = col_sums.count(0)
    single    = col_sums.count(1)
    multi     = K - uncovered - single
    print(f"\n  Border points with 0 radars : {uncovered:5d}  ({100*uncovered/K:.1f}%)")
    print(f"  Border points with 1 radar  : {single:5d}  ({100*single/K:.1f}%)")
    print(f"  Border points with 2+ radars: {multi:5d}  ({100*multi/K:.1f}%)")
    return uncovered


# ══════════════════════════════════════════════════════════════
#  STEP 0 — PURE CSP FEASIBILITY CHECK
# ══════════════════════════════════════════════════════════════

def step0_feasibility(C_mat, N, K, M, L, C_cool, B, time_limit):
    banner("STEP 0 — Feasibility Check (pure CSP)",
           "Can every point be covered at every slot?")

    model = cp_model.CpModel()
    X = [[model.NewBoolVar(f"x_{i}_{t}") for t in range(M)]
         for i in range(N)]

    for j in range(K):
        for t in range(M):
            model.Add(sum(C_mat[i][j] * X[i][t] for i in range(N)) >= 1)

    add_cooling_and_energy(model, X, N, M, L, C_cool, B)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)

    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    if feasible:
        print("  Result : FEASIBLE — a perfect schedule exists.")
        print("  Action : Proceeding to Phase 2 (minimise overlap).")
    else:
        print("  Result : INFEASIBLE — full coverage is impossible.")
        print("  Action : Falling back to Phase 1 (minimise void).")

    return "FEASIBLE" if feasible else "INFEASIBLE"


# ══════════════════════════════════════════════════════════════
#  PHASE 1 — MINIMISE VOID
# ══════════════════════════════════════════════════════════════

def phase1_min_void(C_mat, N, K, M, L, C_cool, B, time_limit):
    banner("PHASE 1 — Minimise Void",
           "Full coverage impossible. Finding minimum-void schedule.")

    model = cp_model.CpModel()
    X = [[model.NewBoolVar(f"x_{i}_{t}") for t in range(M)]
         for i in range(N)]
    V = [[model.NewBoolVar(f"v_{j}_{t}") for t in range(M)]
         for j in range(K)]

    for j in range(K):
        for t in range(M):
            cov = sum(C_mat[i][j] * X[i][t] for i in range(N))
            model.Add(cov >= 1 - V[j][t])

    add_cooling_and_energy(model, X, N, M, L, C_cool, B)
    model.Minimize(sum(V[j][t] for j in range(K) for t in range(M)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)
    label  = status_label(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"  Solver status: {label}")
        return {"phase": 1, "status": label}

    sched      = [[solver.Value(X[i][t]) for t in range(M)] for i in range(N)]
    void_mat   = [[solver.Value(V[j][t]) for t in range(M)] for j in range(K)]
    total_void = sum(void_mat[j][t] for j in range(K) for t in range(M))
    total_on   = sum(sched[i][t]    for i in range(N) for t in range(M))
    void_pct   = 100 * total_void / (K * M) if K * M > 0 else 0

    print(f"\n  Solver status   : {label}")
    print(f"  Total void      : {total_void}  space-time cells uncovered")
    print(f"  Void density    : {void_pct:.2f}%  of ({K} pts × {M} slots)")
    print(f"  Total ON-slots  : {total_on}")

    return {
        "phase":       1,
        "status":      label,
        "total_void":  total_void,
        "void_pct":    void_pct,
        "total_on":    total_on,
        "schedule":    sched,
        "void_matrix": void_mat,
    }


# ══════════════════════════════════════════════════════════════
#  PHASE 2 — MINIMISE OVERLAP
# ══════════════════════════════════════════════════════════════

def phase2_min_overlap(C_mat, N, K, M, L, C_cool, B, time_limit):
    banner("PHASE 2 — Minimise Overlap",
           "Hard coverage enforced. Minimising redundant radar coverage.")

    model = cp_model.CpModel()
    X = [[model.NewBoolVar(f"x_{i}_{t}") for t in range(M)]
         for i in range(N)]
    O = [[model.NewIntVar(0, N - 1, f"o_{j}_{t}") for t in range(M)]
         for j in range(K)]

    for j in range(K):
        for t in range(M):
            cov = sum(C_mat[i][j] * X[i][t] for i in range(N))
            model.Add(cov >= 1)
            model.Add(O[j][t] >= cov - 1)

    add_cooling_and_energy(model, X, N, M, L, C_cool, B)
    model.Minimize(sum(O[j][t] for j in range(K) for t in range(M)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)
    label  = status_label(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"  Solver status: {label}")
        return {"phase": 2, "status": label}

    sched    = [[solver.Value(X[i][t]) for t in range(M)] for i in range(N)]
    over_mat = [[solver.Value(O[j][t]) for t in range(M)] for j in range(K)]
    total_ov = sum(over_mat[j][t] for j in range(K) for t in range(M))
    total_on = sum(sched[i][t]    for i in range(N) for t in range(M))
    ov_den   = total_ov / (K * M) if K * M > 0 else 0

    print(f"\n  Solver status   : {label}")
    print(f"  Total overlap   : {total_ov}  extra radar-point coverages")
    print(f"  Overlap density : {ov_den:.4f}  extra radars per (point × slot)")
    print(f"  Total ON-slots  : {total_on}")
    print(f"  Void            : 0  (hard constraint enforced)")

    return {
        "phase":          2,
        "status":         label,
        "total_overlap":  total_ov,
        "overlap_density": ov_den,
        "total_void":     0,
        "total_on":       total_on,
        "schedule":       sched,
        "overlap_matrix": over_mat,
    }


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "═" * 60)
    print("  SCHEDULER v1.2  —  Two-Phase Radar Scheduler")
    print("═" * 60)

    # ── Load matrix ───────────────────────────────────────────
    print(f"\n[1] Loading {MATRIX_FILE} ...")
    with open(MATRIX_FILE) as f:
        C_mat = json.load(f)

    N = len(C_mat)
    K = len(C_mat[0])
    M = NUM_TIME_SLOTS

    # ── Load metadata ─────────────────────────────────────────
    meta = None
    try:
        with open(META_FILE) as f:
            meta = json.load(f)
        print(f"    Loaded radar metadata: {len(meta)} radars")
    except Exception:
        print("    (No metadata file — using generic radar names)")

    # ── Derive budgets ────────────────────────────────────────
    if meta:
        BUDGETS = assign_budgets(meta, M)
    else:
        BUDGETS = [DEFAULT_BUDGET] * N

    # ── Parameter summary ─────────────────────────────────────
    print(f"\n  N = {N} radars")
    print(f"  K = {K} boundary points")
    print(f"  M = {M} time slots")
    print(f"  L = {L},  C = {C_COOL}  (window = {L + C_COOL})")
    budget_counts = {}
    for b in BUDGETS:
        budget_counts[b] = budget_counts.get(b, 0) + 1
    print(f"  B = {dict(sorted(budget_counts.items()))}  (budget → radar count)")
    print(f"  Time limit per phase: {TIME_LIMIT}s")

    # Validate budget list
    if len(BUDGETS) != N:
        raise ValueError(
            f"BUDGETS has {len(BUDGETS)} entries but N={N} radars. "
            "Check assign_budgets() or BUDGET_BY_TYPE."
        )

    # ── Coverage stats ────────────────────────────────────────
    uncovered_pts = print_coverage_stats(C_mat, N, K)

    if uncovered_pts == K:
        raise RuntimeError(
            "Every boundary point is uncoverable — check radar ranges and positions."
        )

    # ── Step 0 ────────────────────────────────────────────────
    result = step0_feasibility(C_mat, N, K, M, L, C_COOL, BUDGETS, TIME_LIMIT)

    # ── Branch ────────────────────────────────────────────────
    if result == "FEASIBLE":
        output = phase2_min_overlap(C_mat, N, K, M, L, C_COOL, BUDGETS, TIME_LIMIT)
    else:
        output = phase1_min_void(C_mat, N, K, M, L, C_COOL, BUDGETS, TIME_LIMIT)

    # ── Print schedule ────────────────────────────────────────
    if "schedule" in output:
        print_schedule(output["schedule"], N, M, BUDGETS, meta)

    # ── Final summary ─────────────────────────────────────────
    banner("FINAL SUMMARY")
    print(f"  Phase run       : {output['phase']}")
    print(f"  Solver status   : {output['status']}")
    print(f"  N radars        : {N}")
    print(f"  K boundary pts  : {K}")
    print(f"  M time slots    : {M}")
    print(f"  Total ON-slots  : {output.get('total_on', 'N/A')}")
    if output["phase"] == 1:
        print(f"  Total void      : {output.get('total_void', 'N/A')}")
        print(f"  Void density    : {output.get('void_pct', 'N/A'):.2f}%")
    else:
        print(f"  Total overlap   : {output.get('total_overlap', 'N/A')}")
        print(f"  Overlap density : {output.get('overlap_density', 0):.4f}")
        print(f"  Total void      : 0 (guaranteed by hard constraint)")
    print()
