"""
scheduler_1.2.py  —  Two-Phase CSP / ILP Radar Scheduler
=========================================================

Changes over v1.1:
  1. BUDGETS and cooling (L, C) are no longer hardcoded or name-substring-
     matched — they are loaded per-radar-type from radar_type_config.json
     (produced by optimize_placement.py), looked up by an EXACT match on
     each radar's "type" field (via assign_budgets_and_cooling()). Energy
     budget, cooling_L, and cooling_C are now genuinely per-type: different
     radar types can have different cooling windows, not just different
     budgets. Missing radar_meta.json or radar_type_config.json is now a
     hard RuntimeError at startup -- there is no hardcoded fallback.
  2. NUM_TIME_SLOTS (M) is still editable — default stays 24. Hand-edit
     this yourself; it is not sourced from any JSON file.
  3. Prints coverage statistics before solving (how many points each radar
     covers, what fraction of the border is reachable at all).
  4. Minor: status summary prints void % and overlap density for easier
     comparison across runs.

Algorithm is unchanged — see scheduler_1.1.py for full documentation.

Input files (all REQUIRED — no hardcoded fallback if missing):
  c_matrix.json          — built by build_matrix_1.3.py
  radar_meta.json        — built by build_matrix_1.3.py
  radar_type_config.json — built by optimize_placement.py (per-type
                            energy_budget/cooling_L/cooling_C, looked up
                            by exact match on each radar's "type" field)
"""

import json
import sys
from ortools.sat.python import cp_model

# Console output below uses box-drawing/checkmark characters -- force
# UTF-8 on stdout so this doesn't crash under a plain Windows console
# (cp1252), which can't encode them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

MATRIX_FILE      = "c_matrix.json"
META_FILE        = "radar_meta.json"
TYPE_CONFIG_FILE = "radar_type_config.json"

NUM_TIME_SLOTS = 24   # M — 24 hourly slots = one full day
                      # Hand-edit this yourself; not sourced from any JSON
                      # file and not tied to region/radar type.

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


def load_type_config(path):
    """
    Loads radar_type_config.json (schema produced by optimize_placement.py's
    write_type_config_json()): a dict keyed by type name, each value holding
    count/range_m/h_ant/theta_min/theta_max/az_halfwidth/energy_budget/
    cooling_L/cooling_C. Only energy_budget/cooling_L/cooling_C are consumed
    here.
    """
    with open(path) as f:
        type_config = json.load(f)
    if not isinstance(type_config, dict):
        raise RuntimeError(
            f"{path} must contain a JSON object keyed by type name "
            f"(got {type(type_config).__name__})."
        )
    return type_config


def assign_budgets_and_cooling(meta, type_config, M):
    """
    Derive per-radar energy_budget/cooling_L/cooling_C from
    radar_type_config.json via an EXACT lookup on each radar's "type" field
    (not a substring match on "name"). Also enforces budget <= M.

    Explicitly casts to int -- energy_budget arrives as a JSON float from
    optimize_placement.py's wizard (collected via a float prompt), but
    CP-SAT constraint bounds require int.

    Returns three parallel per-radar lists: (budgets, L_list, C_list).
    """
    budgets, L_list, C_list = [], [], []
    for r in meta:
        rtype = r["type"]
        if rtype not in type_config:
            raise RuntimeError(
                f"Radar '{r['name']}' has type '{rtype}', which is not a key in "
                f"{TYPE_CONFIG_FILE} (available types: {sorted(type_config.keys())}). "
                f"{META_FILE} and {TYPE_CONFIG_FILE} are inconsistent with each "
                "other -- they must come from the same optimize_placement.py run. "
                "Regenerate both together."
            )
        cfg = type_config[rtype]
        budgets.append(min(int(round(cfg["energy_budget"])), M))
        L_list.append(int(cfg["cooling_L"]))
        C_list.append(int(cfg["cooling_C"]))
    return budgets, L_list, C_list


def add_cooling_and_energy(model, X, N, M, L_list, C_list, B):
    """
    Cooling  : per-radar rolling-window ON cap -- for radar i, in any window
               of (L_list[i] + C_list[i]) consecutive slots, X may be ON for
               at most L_list[i] of them. Now genuinely per-radar/per-type
               (was a single global L/C_COOL applied uniformly to every
               radar regardless of type).
    Energy   : Σ_t X[i,t]  ≤  B[i]  ∀ i
    """
    for i in range(N):
        window = L_list[i] + C_list[i]
        if window > 1:
            for t in range(M):
                end = min(t + window, M)
                model.Add(sum(X[i][k] for k in range(t, end)) <= L_list[i])
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

def step0_feasibility(C_mat, N, K, M, L_list, C_list, B, time_limit):
    banner("STEP 0 — Feasibility Check (pure CSP)",
           "Can every point be covered at every slot?")

    model = cp_model.CpModel()
    X = [[model.NewBoolVar(f"x_{i}_{t}") for t in range(M)]
         for i in range(N)]

    for j in range(K):
        for t in range(M):
            model.Add(sum(C_mat[i][j] * X[i][t] for i in range(N)) >= 1)

    add_cooling_and_energy(model, X, N, M, L_list, C_list, B)

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

def phase1_min_void(C_mat, N, K, M, L_list, C_list, B, time_limit):
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

    add_cooling_and_energy(model, X, N, M, L_list, C_list, B)
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

def phase2_min_overlap(C_mat, N, K, M, L_list, C_list, B, time_limit):
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

    add_cooling_and_energy(model, X, N, M, L_list, C_list, B)
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

    # ── Load metadata (now REQUIRED — no more "generic radar names"
    #    fallback, since budgets/cooling can no longer be derived without
    #    it) ──────────────────────────────────────────────────
    try:
        with open(META_FILE) as f:
            meta = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(
            f"{META_FILE} not found. This scheduler now requires {META_FILE} "
            f"(for radar type lookup) and {TYPE_CONFIG_FILE} (for per-type "
            "energy_budget/cooling_L/cooling_C) -- there is no more hardcoded "
            "fallback. Run build_matrix_1.3.py first to produce it."
        )
    print(f"    Loaded radar metadata: {len(meta)} radars")

    # ── Load per-type config ──────────────────────────────────
    try:
        type_config = load_type_config(TYPE_CONFIG_FILE)
    except FileNotFoundError:
        raise RuntimeError(
            f"{TYPE_CONFIG_FILE} not found. This scheduler now requires "
            f"{TYPE_CONFIG_FILE} for per-type energy_budget/cooling_L/cooling_C "
            "-- there is no more hardcoded fallback. Run optimize_placement.py "
            "first to produce it."
        )
    print(f"    Loaded config for {len(type_config)} type(s): {sorted(type_config.keys())}")

    # ── Derive budgets + cooling ───────────────────────────────
    BUDGETS, L_LIST, C_LIST = assign_budgets_and_cooling(meta, type_config, M)

    # ── Parameter summary ─────────────────────────────────────
    print(f"\n  N = {N} radars")
    print(f"  K = {K} boundary points")
    print(f"  M = {M} time slots")
    cooling_counts = {}
    for l_val, c_val in zip(L_LIST, C_LIST):
        cooling_counts[(l_val, c_val)] = cooling_counts.get((l_val, c_val), 0) + 1
    cooling_summary = ", ".join(
        f"(L={l_val},C={c_val})×{count}"
        for (l_val, c_val), count in sorted(cooling_counts.items())
    )
    print(f"  Cooling (L,C) by radar: {cooling_summary}  (per-type, from {TYPE_CONFIG_FILE})")
    budget_counts = {}
    for b in BUDGETS:
        budget_counts[b] = budget_counts.get(b, 0) + 1
    print(f"  B = {dict(sorted(budget_counts.items()))}  (budget → radar count)")
    print(f"  Time limit per phase: {TIME_LIMIT}s")

    # Validate budget/cooling lists
    if len(BUDGETS) != N or len(L_LIST) != N or len(C_LIST) != N:
        raise ValueError(
            f"BUDGETS/L_LIST/C_LIST have {len(BUDGETS)}/{len(L_LIST)}/{len(C_LIST)} "
            f"entries but N={N} radars. Check assign_budgets_and_cooling() or "
            f"{TYPE_CONFIG_FILE}."
        )

    # ── Coverage stats ────────────────────────────────────────
    uncovered_pts = print_coverage_stats(C_mat, N, K)

    if uncovered_pts == K:
        raise RuntimeError(
            "Every boundary point is uncoverable — check radar ranges and positions."
        )

    # ── Step 0 ────────────────────────────────────────────────
    result = step0_feasibility(C_mat, N, K, M, L_LIST, C_LIST, BUDGETS, TIME_LIMIT)

    # ── Branch ────────────────────────────────────────────────
    if result == "FEASIBLE":
        output = phase2_min_overlap(C_mat, N, K, M, L_LIST, C_LIST, BUDGETS, TIME_LIMIT)
    else:
        output = phase1_min_void(C_mat, N, K, M, L_LIST, C_LIST, BUDGETS, TIME_LIMIT)

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

    with open("schedule_output.json", "w") as f:
        json.dump(output, f)
