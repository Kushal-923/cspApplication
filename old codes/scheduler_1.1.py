"""
scheduler.py  —  Two-Phase CSP / ILP Radar Scheduler
=====================================================

Fixes over the previous (Gemini) version:
  1. Phase 0 is a pure feasibility check (separate model, no objective).
  2. Phase 1 (min void) and Phase 2 (min overlap) are SEPARATE models.
     OR-Tools CP-SAT does not support changing the objective mid-solve
     reliably; the Gemini version was calling solver.Solve() twice on the
     same model instance, which is unsupported and gives wrong results.
  3. Cooling / duty-cycle constraint is added:
       In any rolling window of (L + C) consecutive slots,
       radar i may be ON for at most L of them.
  4. O variable upper bound is N-1, not N.
  5. Step 0 → Phase 2 path is correctly taken when full coverage exists.

Algorithm flow (matches your handwritten notes):
  Step 0  pure CSP feasibility  →  FEASIBLE   → Phase 2
                                →  INFEASIBLE → Phase 1

Input files:
  c_matrix.json    — built by build_matrix.py
  radar_meta.json  — built by build_matrix.py  (optional, for display)
"""

import json
from ortools.sat.python import cp_model

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION  — edit these to match your scenario
# ══════════════════════════════════════════════════════════════

MATRIX_FILE = "c_matrix.json"
META_FILE   = "radar_meta.json"   # set to None if not present

NUM_TIME_SLOTS = 24   # M  — 24 hourly time slots (one full day)

# Cooling constraint parameters
#   In any window of (L + C_COOL) slots, radar may be ON at most L slots.
#   Example: L=3, C_COOL=2  →  window=5, at most 3 ON per window
L      = 3    # max ON-slots inside any rolling window
C_COOL = 2    # mandatory rest-slots inside the same window

# Energy budgets B[i] — max total ON-slots for each radar over all M slots
# Must have one entry per radar.  Radars are in the same order as c_matrix.json.
# Small=20, Medium=16, Large=12, Small=20, Medium=16  (for the 5-radar setup)
BUDGETS = [20, 16, 12, 20, 16]

# Solver time limit per phase (seconds)
TIME_LIMIT = 60.0

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


def add_cooling_and_energy(model, X, N, M, L, C_cool, B):
    """
    Cooling  : Σ_{k=t}^{min(t+L+C-1, M-1)} X[i,k]  ≤  L   ∀ i, t
    Energy   : Σ_t X[i,t]  ≤  B[i]                         ∀ i
    """
    window = L + C_cool
    if window > 1:
        for i in range(N):
            for t in range(M):
                end = min(t + window, M)          # cap at M (notes: "put min")
                model.Add(sum(X[i][k] for k in range(t, end)) <= L)

    for i in range(N):
        model.Add(sum(X[i][t] for t in range(M)) <= B[i])


def print_schedule(sched, N, M, B, meta):
    print("\n  FINAL SCHEDULE  (1 = ON, 0 = OFF)")
    header = "                  " + " ".join(f"{t:2d}" for t in range(M))
    print(header)
    for i in range(N):
        on_count = sum(sched[i])
        name     = meta[i]["name"] if meta else f"R{i+1}"
        row      = "  ".join(str(sched[i][t]) for t in range(M))
        print(f"  {name:12s}  [{on_count:2d}/{B[i]}]  {row}")


# ══════════════════════════════════════════════════════════════
#  STEP 0 — PURE CSP FEASIBILITY CHECK  (separate model)
# ══════════════════════════════════════════════════════════════

def step0_feasibility(C_mat, N, K, M, L, C_cool, B, time_limit):
    """
    Fresh model, no objective — just asks: does a zero-void schedule exist?
    Returns 'FEASIBLE' or 'INFEASIBLE'.
    """
    banner("STEP 0 — Feasibility Check (pure CSP)",
           "Does a zero-void schedule exist given all constraints?")

    model  = cp_model.CpModel()
    X = [[model.NewBoolVar(f"x_{i}_{t}") for t in range(M)]
         for i in range(N)]

    # Hard covering: every point covered at every slot
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
#  PHASE 1 — MINIMISE VOID  (fresh model)
# ══════════════════════════════════════════════════════════════

def phase1_min_void(C_mat, N, K, M, L, C_cool, B, time_limit):
    """
    Full coverage impossible.  Introduce V[j,t] ∈ {0,1} void variables.

    Relaxed covering : Σ_i C[i,j]·X[i,t]  ≥  1 − V[j,t]    ∀ j, t
    Objective        : min  Σ_{j,t} V[j,t]
    """
    banner("PHASE 1 — Minimise Void",
           "Full coverage impossible. Finding minimum-void schedule.")

    model  = cp_model.CpModel()
    X = [[model.NewBoolVar(f"x_{i}_{t}") for t in range(M)]
         for i in range(N)]
    V = [[model.NewBoolVar(f"v_{j}_{t}") for t in range(M)]
         for j in range(K)]

    # Relaxed covering
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
    print(f"  Void %          : {void_pct:.2f}% of ({K} pts × {M} slots)")
    print(f"  Total ON-slots  : {total_on}")

    return {
        "phase":       1,
        "status":      label,
        "total_void":  total_void,
        "total_on":    total_on,
        "schedule":    sched,
        "void_matrix": void_mat,
    }


# ══════════════════════════════════════════════════════════════
#  PHASE 2 — MINIMISE OVERLAP  (fresh model, hard covering)
# ══════════════════════════════════════════════════════════════

def phase2_min_overlap(C_mat, N, K, M, L, C_cool, B, time_limit):
    """
    Full coverage guaranteed as hard constraint.
    O[j,t] ∈ {0 … N-1} counts extra radars covering point j at slot t.

    Hard covering    : Σ_i C[i,j]·X[i,t]  ≥  1             ∀ j, t
    Overlap lower bnd: O[j,t]              ≥  Σ_i C[i,j]·X[i,t] − 1
    Objective        : min  Σ_{j,t} O[j,t]
    """
    banner("PHASE 2 — Minimise Overlap",
           "Hard coverage enforced. Minimising redundant radar coverage.")

    model  = cp_model.CpModel()
    X = [[model.NewBoolVar(f"x_{i}_{t}") for t in range(M)]
         for i in range(N)]
    # O upper bound is N-1 (at most N-1 *extra* radars)
    O = [[model.NewIntVar(0, N - 1, f"o_{j}_{t}") for t in range(M)]
         for j in range(K)]

    # Hard covering
    for j in range(K):
        for t in range(M):
            cov = sum(C_mat[i][j] * X[i][t] for i in range(N))
            model.Add(cov >= 1)

    # Overlap lower bound
    for j in range(K):
        for t in range(M):
            cov = sum(C_mat[i][j] * X[i][t] for i in range(N))
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

    sched      = [[solver.Value(X[i][t]) for t in range(M)] for i in range(N)]
    over_mat   = [[solver.Value(O[j][t]) for t in range(M)] for j in range(K)]
    total_ov   = sum(over_mat[j][t] for j in range(K) for t in range(M))
    total_on   = sum(sched[i][t]    for i in range(N) for t in range(M))

    print(f"\n  Solver status   : {label}")
    print(f"  Total overlap   : {total_ov}  extra radar-point coverages")
    print(f"  Total ON-slots  : {total_on}")
    print(f"  Void            : 0  (hard constraint enforced)")

    return {
        "phase":          2,
        "status":         label,
        "total_overlap":  total_ov,
        "total_void":     0,
        "total_on":       total_on,
        "schedule":       sched,
        "overlap_matrix": over_mat,
    }


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Load matrix ───────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  SCHEDULER.PY  —  Two-Phase Radar Scheduler")
    print("═" * 60)

    print(f"\n[1] Loading coverage matrix from {MATRIX_FILE} ...")
    with open(MATRIX_FILE) as f:
        C_mat = json.load(f)

    N = len(C_mat)
    K = len(C_mat[0])
    M = NUM_TIME_SLOTS

    # Load optional metadata for display
    meta = None
    try:
        with open(META_FILE) as f:
            meta = json.load(f)
        print(f"    Loaded radar metadata from {META_FILE}")
    except Exception:
        print(f"    (No metadata file — using generic radar names)")

    print(f"\n  N = {N} radars")
    print(f"  K = {K} boundary points")
    print(f"  M = {M} time slots")
    print(f"  L = {L},  C = {C_COOL}  (window = {L + C_COOL})")
    print(f"  B = {BUDGETS}")

    # Validate budget list length
    if len(BUDGETS) != N:
        raise ValueError(
            f"BUDGETS has {len(BUDGETS)} entries but there are {N} radars. "
            "Edit the BUDGETS list in the configuration section."
        )

    # ── Step 0: feasibility ───────────────────────────────────
    result = step0_feasibility(C_mat, N, K, M, L, C_COOL, BUDGETS, TIME_LIMIT)

    # ── Branch ────────────────────────────────────────────────
    if result == "FEASIBLE":
        output = phase2_min_overlap(C_mat, N, K, M, L, C_COOL, BUDGETS, TIME_LIMIT)
    else:
        output = phase1_min_void(C_mat, N, K, M, L, C_COOL, BUDGETS, TIME_LIMIT)

    # ── Print schedule ────────────────────────────────────────
    if "schedule" in output:
        print_schedule(output["schedule"], N, M, BUDGETS, meta)

    # ── Summary ───────────────────────────────────────────────
    banner("SUMMARY")
    print(f"  Phase run       : {output['phase']}")
    print(f"  Solver status   : {output['status']}")
    print(f"  Total ON-slots  : {output.get('total_on', 'N/A')}")
    if output["phase"] == 1:
        print(f"  Total void      : {output.get('total_void', 'N/A')}")
    else:
        print(f"  Total overlap   : {output.get('total_overlap', 'N/A')}")
        print(f"  Total void      : 0 (guaranteed)")
    print()
