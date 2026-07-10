"""
Radar Scheduling — Two-Phase CSP/ILP Solver
============================================

Based on handwritten formulation notes (Pages 1–5).

ALGORITHM FLOW
──────────────
Step 0 · Pure CSP feasibility check
          Variables : X[i,t] ∈ {0,1}
          Constraints: Covering, Cooling, Energy
          → FEASIBLE   : go to Phase 2
          → INFEASIBLE : go to Phase 1 (fallback)

Phase 1 · Minimise Void (fallback when full coverage impossible)
          New variable : V[j,t] ∈ {0,1}   (1 = point j uncovered at t)
          Relaxed covering : Σ_i C[i,j]·X[i,t]  ≥  1 - V[j,t]
          Objective        : min Σ_{t,j} V[j,t]

Phase 2 · Minimise Overlap (only when full coverage is achievable)
          New variable : O[j,t] ∈ {0,…,N-1}  (extra radars on same point)
          Hard covering   : Σ_i C[i,j]·X[i,t]  ≥  1
          Overlap def     : O[j,t]              ≥  Σ_i C[i,j]·X[i,t] - 1
          Objective       : min Σ_{t,j} O[j,t]

CONSTRAINTS (shared by all phases)
───────────────────────────────────
Covering : Σ_i C[i,j]·X[i,t] ≥ 1  ∀ j,t          (hard or softened)
Cooling  : Σ_{k=t}^{t+L+C-1} X[i,k] ≤ L  ∀ i,t   (duty-cycle window)
Energy   : Σ_t X[i,t] ≤ B[i]               ∀ i

NOTATION
────────
N        number of radars
K        number of boundary points
M        number of time slots  (notes: M = 248)
L        max ON-slots inside any rolling window of L+C slots
C        cool-down slots inside the same window
B[i]     energy budget (max total ON-slots) for radar i
C_mat    coverage matrix  C[i][j] ∈ {0,1}
X[i,t]   decision variable — radar i ON at time t
V[j,t]   void variable    — point j uncovered at time t
O[j,t]   overlap variable — extra radars covering point j at time t
"""

import math
from ortools.sat.python import cp_model


# ══════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════

def build_boundary_points(K: int) -> list[tuple[float, float]]:
    """
    K evenly-spaced points on the unit circle representing the
    monitored boundary (P_1 … P_K in the notes).
    """
    return [
        (math.cos(2 * math.pi * k / K), math.sin(2 * math.pi * k / K))
        for k in range(K)
    ]


def build_coverage_matrix(
    radars: list[tuple[float, float]],
    pts:    list[tuple[float, float]],
    R:      float
) -> list[list[int]]:
    """
    C[i][j] = 1  iff  distance(radar_i, point_j) ≤ R_i.

    From notes:
      C_{ij} = 1  if  √((x_i−x_j)² + (y_i−y_j)²) ≤ r_i
               0  otherwise
    """
    return [
        [1 if math.hypot(rx - px, ry - py) <= R else 0
         for (px, py) in pts]
        for (rx, ry) in radars
    ]


# ══════════════════════════════════════════════════════════════
#  SHARED CONSTRAINT BUILDER
# ══════════════════════════════════════════════════════════════

def add_cooling_and_energy(
    model:  cp_model.CpModel,
    X:      list[list],
    N:      int,
    M:      int,
    L:      int,
    C_cool: int,
    B:      list[int]
) -> None:
    """
    Adds cooling and energy constraints to any model phase.

    Cooling (duty-cycle window, from notes page 1 & 2):
      Σ_{k=t}^{min(t+L+C−1, M−1)} X[i,k] ≤ L    ∀ i, t
      Notes: "goes out of bounds so put min" — upper index capped at M.

    Energy budget (from notes page 1):
      Σ_t X[i,t] ≤ B[i]    ∀ i
    """
    window = L + C_cool

    # ── Cooling ──────────────────────────────────────────────
    if window > 1:
        for i in range(N):
            for t in range(M):
                end = min(t + window, M)   # cap at M as noted
                model.Add(
                    sum(X[i][k] for k in range(t, end)) <= L
                )

    # ── Energy budget ─────────────────────────────────────────
    for i in range(N):
        model.Add(sum(X[i][t] for t in range(M)) <= B[i])


# ══════════════════════════════════════════════════════════════
#  STEP 0 — FEASIBILITY CHECK  (pure CSP)
# ══════════════════════════════════════════════════════════════

def feasibility_check(
    C_mat:  list[list[int]],
    N:      int,
    K:      int,
    M:      int,
    L:      int,
    C_cool: int,
    B:      list[int],
    time_limit: float = 30.0,
    verbose: bool = True
) -> str:
    """
    Pure CSP: does there exist X[i,t] satisfying all hard constraints?

    Hard covering:  Σ_i C[i,j]·X[i,t]  ≥ 1    ∀ j, t
    + Cooling + Energy

    Returns 'FEASIBLE' or 'INFEASIBLE'.
    """
    if verbose:
        _banner("STEP 0 — Feasibility Check (pure CSP)",
                "Question: does a zero-void schedule exist?")

    model = cp_model.CpModel()
    X = [[model.NewBoolVar(f'x_{i}_{t}') for t in range(M)]
         for i in range(N)]

    # Hard covering — every point covered at every slot
    for j in range(K):
        for t in range(M):
            model.Add(sum(C_mat[i][j] * X[i][t] for i in range(N)) >= 1)

    add_cooling_and_energy(model, X, N, M, L, C_cool, B)

    # No objective — pure feasibility
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)

    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    if verbose:
        if feasible:
            print("  Result : FEASIBLE — full coverage schedule exists!")
            print("  Action : Proceeding to Phase 2 (minimise overlap).\n")
        else:
            print("  Result : INFEASIBLE — cannot cover all points.")
            print("  Action : Falling back to Phase 1 (minimise void).\n")

    return 'FEASIBLE' if feasible else 'INFEASIBLE'


# ══════════════════════════════════════════════════════════════
#  PHASE 1 — MINIMISE VOID  (fallback)
# ══════════════════════════════════════════════════════════════

def phase1_min_void(
    C_mat:  list[list[int]],
    N:      int,
    K:      int,
    M:      int,
    L:      int,
    C_cool: int,
    B:      list[int],
    time_limit: float = 60.0,
    verbose: bool = True
) -> dict:
    """
    Full coverage is impossible. Best we can do: minimise blind spots.

    New variable  : V[j,t] ∈ {0,1}
                    V[j,t] = 1  ↔  point j not covered at time t

    Relaxed covering:
      Σ_i C[i,j]·X[i,t]  ≥  1 − V[j,t]    ∀ j, t

    Objective:
      min  Σ_{t=1}^{M} Σ_{j=1}^{K}  V[j,t]

    From notes page 2 & 4.
    """
    if verbose:
        _banner("PHASE 1 — Minimise Void",
                "Full coverage impossible. Finding minimum-void schedule.")

    model = cp_model.CpModel()

    X = [[model.NewBoolVar(f'x_{i}_{t}') for t in range(M)]
         for i in range(N)]
    V = [[model.NewBoolVar(f'v_{j}_{t}') for t in range(M)]
         for j in range(K)]

    # Relaxed covering — void variable absorbs uncovered cells
    for j in range(K):
        for t in range(M):
            cov = sum(C_mat[i][j] * X[i][t] for i in range(N))
            model.Add(cov >= 1 - V[j][t])

    add_cooling_and_energy(model, X, N, M, L, C_cool, B)

    # Objective: minimise total void space-time cells
    model.Minimize(sum(V[j][t] for j in range(K) for t in range(M)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)

    label = _status_label(status)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sched = [[solver.Value(X[i][t]) for t in range(M)] for i in range(N)]
        vmat  = [[solver.Value(V[j][t]) for t in range(M)] for j in range(K)]
        total_void = sum(vmat[j][t] for j in range(K) for t in range(M))
        total_on   = sum(sched[i][t] for i in range(N) for t in range(M))

        if verbose:
            _print_phase1(label, sched, vmat, C_mat, N, K, M, total_void, total_on, B)

        return {
            'phase':       1,
            'status':      label,
            'total_void':  total_void,
            'total_on':    total_on,
            'schedule':    sched,
            'void_matrix': vmat,
        }
    else:
        if verbose:
            print(f"  Status: {label}")
        return {'phase': 1, 'status': label}


# ══════════════════════════════════════════════════════════════
#  PHASE 2 — MINIMISE OVERLAP  (when full coverage is achievable)
# ══════════════════════════════════════════════════════════════

def phase2_min_overlap(
    C_mat:  list[list[int]],
    N:      int,
    K:      int,
    M:      int,
    L:      int,
    C_cool: int,
    B:      list[int],
    time_limit: float = 60.0,
    verbose: bool = True
) -> dict:
    """
    Full coverage is guaranteed as a hard constraint.
    Minimise redundant (overlapping) radar coverage.

    New variable  : O[j,t] ∈ {0, …, N−1}
                    O[j,t] = number of *extra* radars covering point j at t

    From notes page 3 & 5:
      O[j,t] = max(0,  Σ_i C[i,j]·X[i,t] − 1)

    Linearised for ILP (notes: "we have to convert to LP"):
      O[j,t]  ≥  Σ_i C[i,j]·X[i,t] − 1    ∀ j, t
      O[j,t]  ≥  0                          (from domain)
    The minimiser drives O[j,t] to the tight lower bound automatically.

    Objective:
      min  Σ_{t=1}^{M} Σ_{j=1}^{K}  O[j,t]

    Constraints same as Phase 1 but with hard covering (no V variables).
    """
    if verbose:
        _banner("PHASE 2 — Minimise Overlap",
                "Hard coverage enforced. Minimising redundant sensors.")

    model = cp_model.CpModel()

    X = [[model.NewBoolVar(f'x_{i}_{t}') for t in range(M)]
         for i in range(N)]
    # O[j,t] ∈ {0, …, N-1}   (at most N−1 extra radars)
    O = [[model.NewIntVar(0, N - 1, f'o_{j}_{t}') for t in range(M)]
         for j in range(K)]

    # Hard covering — no void allowed
    for j in range(K):
        for t in range(M):
            cov = sum(C_mat[i][j] * X[i][t] for i in range(N))
            model.Add(cov >= 1)

    # Overlap lower bound: O[j,t] ≥ coverage_sum − 1
    for j in range(K):
        for t in range(M):
            cov = sum(C_mat[i][j] * X[i][t] for i in range(N))
            model.Add(O[j][t] >= cov - 1)

    add_cooling_and_energy(model, X, N, M, L, C_cool, B)

    # Objective: minimise total overlap
    model.Minimize(sum(O[j][t] for j in range(K) for t in range(M)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)

    label = _status_label(status)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sched    = [[solver.Value(X[i][t]) for t in range(M)] for i in range(N)]
        omat     = [[solver.Value(O[j][t]) for t in range(M)] for j in range(K)]
        total_ov = sum(omat[j][t] for j in range(K) for t in range(M))
        total_on = sum(sched[i][t] for i in range(N) for t in range(M))

        if verbose:
            _print_phase2(label, sched, omat, C_mat, N, K, M, total_ov, total_on, B)

        return {
            'phase':          2,
            'status':         label,
            'total_overlap':  total_ov,
            'total_void':     0,
            'total_on':       total_on,
            'schedule':       sched,
            'overlap_matrix': omat,
        }
    else:
        if verbose:
            print(f"  Status: {label}")
        return {'phase': 2, 'status': label}


# ══════════════════════════════════════════════════════════════
#  MASTER ENTRY POINT
# ══════════════════════════════════════════════════════════════

def run(
    radars:    list[tuple[float, float]],
    K:         int,
    R:         float,
    M:         int,
    L:         int,
    C_cool:    int,
    B:         list[int],
    time_limit: float = 60.0,
    verbose:   bool = True
) -> dict:
    """
    Full two-phase radar scheduling solver.

    Parameters
    ──────────
    radars     : list of (x, y) radar positions
    K          : number of boundary points on the unit circle
    R          : common sensing radius
    M          : number of time slots  (notes suggest M = 248 for 24h × 5-min slots)
    L          : max ON-slots inside any window of L+C_cool  (cooling param)
    C_cool     : mandatory OFF-slots inside the same window  (cooling param)
    B          : energy budget list — B[i] = max total ON-slots for radar i
    time_limit : seconds given to each ILP solve
    verbose    : print detailed log

    Returns
    ───────
    dict with keys:
      phase, status, schedule, total_on
      + total_void     (Phase 1 / fallback)
      + total_overlap  (Phase 2)
      + void_matrix    (Phase 1 / fallback)
      + overlap_matrix (Phase 2)

    Flow  (from notes page 4)
    ──────────────────────────
    Step 0 → CSP feasibility
      FEASIBLE   → Phase 2 (minimise overlap)
      INFEASIBLE → Phase 1 (minimise void)
    """
    N     = len(radars)
    pts   = build_boundary_points(K)
    C_mat = build_coverage_matrix(radars, pts, R)

    if verbose:
        print(f"\n{'═'*58}")
        print(f"  RADAR CSP SCHEDULER")
        print(f"  N={N} radars  K={K} boundary pts  R={R}")
        print(f"  M={M} slots  |  L={L}  C={C_cool}  B={B}")
        print(f"{'═'*58}")
        _print_coverage_summary(C_mat, N, K)

    # ── Step 0: feasibility check ──────────────────────────────
    feasibility = feasibility_check(C_mat, N, K, M, L, C_cool, B,
                                    time_limit, verbose)

    # ── Branch ────────────────────────────────────────────────
    if feasibility == 'FEASIBLE':
        return phase2_min_overlap(C_mat, N, K, M, L, C_cool, B,
                                  time_limit, verbose)
    else:
        return phase1_min_void(C_mat, N, K, M, L, C_cool, B,
                               time_limit, verbose)


# ══════════════════════════════════════════════════════════════
#  PRINT HELPERS
# ══════════════════════════════════════════════════════════════

def _banner(title: str, subtitle: str = "") -> None:
    print(f"\n{'─'*58}")
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(f"{'─'*58}")


def _status_label(status: int) -> str:
    return {
        cp_model.OPTIMAL:    'OPTIMAL',
        cp_model.FEASIBLE:   'FEASIBLE',
        cp_model.INFEASIBLE: 'INFEASIBLE',
        cp_model.UNKNOWN:    'UNKNOWN',
    }.get(status, 'UNKNOWN')


def _print_coverage_summary(C_mat, N, K):
    print("\n  COVERAGE MATRIX  C[i][j]  (radar × boundary point)")
    header = "        " + " ".join(f"p{j+1:<2}" for j in range(K))
    print(header)
    for i in range(N):
        row = "  ".join(str(C_mat[i][j]) for j in range(K))
        print(f"  R{i+1:2d}:  {row}   [{sum(C_mat[i])}/{K} pts reachable]")
    print()


def _print_schedule(sched, N, M, B):
    """Print ON/OFF schedule compactly."""
    if M <= 20:
        print("\n  SCHEDULE  (1=ON  0=OFF)")
        header = "         " + "  ".join(f"t{t}" for t in range(M))
        print(header)
        for i in range(N):
            row = "   ".join(str(sched[i][t]) for t in range(M))
            on  = sum(sched[i])
            print(f"  R{i+1:2d}:  {row}   [ON={on}/{B[i]}]")
    else:
        # Compact summary for large M
        print("\n  SCHEDULE SUMMARY  (total ON slots per radar)")
        for i in range(N):
            on  = sum(sched[i])
            pct = 100 * on / M
            bar = '█' * int(pct / 5)
            print(f"  R{i+1:2d}: {on:>4}/{B[i]} slots ON  ({pct:5.1f}%)  {bar}")


def _print_phase1(status, sched, vmat, C_mat, N, K, M, total_void, total_on, B):
    print(f"\n  Status          : {status}")
    print(f"  Total void      : {total_void}  space-time cells uncovered")
    print(f"  Total ON slots  : {total_on}")
    void_pct = 100 * total_void / (K * M) if K * M > 0 else 0
    print(f"  Void coverage % : {void_pct:.2f}% of (point × slot) pairs")

    _print_schedule(sched, N, M, B)

    if total_void > 0 and M <= 20:
        print("\n  VOID MATRIX  (only points with voids shown)")
        header = "         " + "  ".join(f"t{t}" for t in range(M))
        print(header)
        for j in range(K):
            v = sum(vmat[j])
            if v > 0:
                row = "   ".join(str(vmat[j][t]) for t in range(M))
                print(f"  p{j+1:02d}: {row}   [voids={v}]")
    elif total_void == 0:
        print("\n  ✓ Zero void — all points covered despite infeasibility signal!")
    else:
        # Large M: summarise void per point
        print("\n  VOID SUMMARY (points with voids)")
        for j in range(K):
            v = sum(vmat[j])
            if v > 0:
                pct = 100 * v / M
                print(f"    p{j+1:02d}: {v}/{M} slots void  ({pct:.1f}%)")

    print("\n  COVERAGE MATRIX (reference)")
    _print_coverage_summary(C_mat, N, K)


def _print_phase2(status, sched, omat, C_mat, N, K, M, total_ov, total_on, B):
    print(f"\n  Status          : {status}")
    print(f"  Total overlap   : {total_ov}  extra radar-point coverages")
    print(f"  Total ON slots  : {total_on}")
    print(f"  Void            : 0  (hard constraint enforced)")

    _print_schedule(sched, N, M, B)

    if total_ov > 0 and M <= 20:
        print("\n  OVERLAP MATRIX  (only points with overlap shown)")
        header = "         " + "  ".join(f"t{t}" for t in range(M))
        print(header)
        for j in range(K):
            o = sum(omat[j])
            if o > 0:
                row = "   ".join(str(omat[j][t]) for t in range(M))
                print(f"  p{j+1:02d}: {row}   [total={o}]")
    elif total_ov == 0:
        print("\n  ✓ Zero overlap — each point covered by exactly one radar!")
    else:
        print("\n  OVERLAP SUMMARY (points with overlap)")
        for j in range(K):
            o = sum(omat[j])
            if o > 0:
                pct = 100 * o / M
                print(f"    p{j+1:02d}: overlapped in {o}/{M} slots  ({pct:.1f}%)")

    print("\n  COVERAGE MATRIX (reference)")
    _print_coverage_summary(C_mat, N, K)


# ══════════════════════════════════════════════════════════════
#  USER INPUT HELPERS
# ══════════════════════════════════════════════════════════════

def _prompt_int(prompt: str, min_val: int = 1) -> int:
    """Keep asking until the user enters a valid integer >= min_val."""
    while True:
        try:
            val = int(input(prompt).strip())
            if val < min_val:
                print(f"  ✗ Must be at least {min_val}. Try again.")
            else:
                return val
        except ValueError:
            print("  ✗ Please enter a whole number.")


def _prompt_float(prompt: str, min_val: float = 0.0) -> float:
    """Keep asking until the user enters a valid number > min_val (int or float accepted)."""
    while True:
        try:
            val = float(input(prompt).strip())
            if val <= min_val:
                print(f"  ✗ Must be greater than {min_val}. Try again.")
            else:
                return val
        except ValueError:
            print("  ✗ Please enter a number (e.g. 1.3).")


def _prompt_radar_coords(N: int) -> list[tuple[float, float]]:
    """
    Ask user how they want to supply radar positions, then collect them.

    Option A — manual: enter (x, y) for each radar one by one.
    Option B — evenly spaced on unit circle (auto-generated).
    """
    print("\n  How do you want to place the radars?")
    print("    [1] Enter each (x, y) coordinate manually")
    print("    [2] Place them evenly on the unit circle (auto)")
    while True:
        choice = input("  Choice (1 or 2): ").strip()
        if choice in ("1", "2"):
            break
        print("  ✗ Enter 1 or 2.")

    if choice == "2":
        radars = [
            (math.cos(2*math.pi*k/N), math.sin(2*math.pi*k/N))
            for k in range(N)
        ]
        print(f"  ✓ Auto-placed {N} radars evenly on the unit circle.")
        return radars

    # Manual entry
    radars = []
    for i in range(N):
        while True:
            try:
                raw = input(f"    Radar {i+1} — enter x y (space-separated): ").strip()
                x, y = map(float, raw.split())
                radars.append((x, y))
                break
            except ValueError:
                print("  ✗ Enter two numbers separated by a space, e.g.  1.0 0.5")
    return radars


def _prompt_budgets(N: int, M: int) -> list[int]:
    """
    Ask whether all radars share the same budget or each has its own.
    Budget is validated to be in [1, M].
    """
    print(f"\n  Energy budget B[i] = max ON-slots per radar  (1 – {M})")
    print("    [1] Same budget for all radars")
    print("    [2] Enter a budget for each radar individually")
    while True:
        choice = input("  Choice (1 or 2): ").strip()
        if choice in ("1", "2"):
            break
        print("  ✗ Enter 1 or 2.")

    if choice == "1":
        b = _prompt_int(f"  Budget for all radars (1–{M}): ", min_val=1)
        if b > M:
            print(f"  ⚠ Budget capped at M={M}.")
            b = M
        return [b] * N

    budgets = []
    for i in range(N):
        b = _prompt_int(f"  Budget for radar {i+1} (1–{M}): ", min_val=1)
        if b > M:
            print(f"  ⚠ Budget capped at M={M}.")
            b = M
        budgets.append(b)
    return budgets


def collect_inputs() -> dict:
    """
    Interactive CLI to collect all parameters needed by run().

    Returns a dict with keys:
      radars, K, R, M, L, C_cool, B, time_limit
    """
    print("\n" + "═"*58)
    print("  RADAR CSP SCHEDULER — Parameter Setup")
    print("═"*58)

    # ── Number of radars ──────────────────────────────────────
    N = _prompt_int("\n  Number of radars N: ", min_val=1)

    # ── Radar positions ───────────────────────────────────────
    radars = _prompt_radar_coords(N)

    # ── Sensing radius ────────────────────────────────────────
    R = _prompt_float("\n  Sensing radius R (e.g. 1.3): ", min_val=0.0)

    # ── Boundary points ───────────────────────────────────────
    K = _prompt_int("\n  Number of boundary points K (e.g. 12): ", min_val=1)

    # ── Time slots ────────────────────────────────────────────
    print("\n  Number of time slots M")
    print("  (e.g. 248 = 24 h at 5-min resolution, 48 = 24 h at 30-min)")
    M = _prompt_int("  M: ", min_val=1)

    # ── Cooling parameters ────────────────────────────────────
    print("\n  Cooling constraint: in any window of L+C consecutive slots,")
    print("  a radar may be ON for at most L slots and must rest for C slots.")
    L      = _prompt_int("  L (max ON-slots in window, e.g. 3): ", min_val=1)
    C_cool = _prompt_int("  C (cool-down slots in same window, e.g. 1, enter 0 for none): ",
                         min_val=0)

    # ── Energy budgets ────────────────────────────────────────
    B = _prompt_budgets(N, M)

    # ── Solver time limit ─────────────────────────────────────
    print(f"\n  Solver time limit per phase in seconds")
    print(f"  (recommended: ≥60 for M>50, ≥120 for M>100)")
    while True:
        try:
            tl = float(input("  Time limit (seconds, e.g. 60): ").strip())
            if tl <= 0:
                print("  ✗ Must be positive.")
            else:
                break
        except ValueError:
            print("  ✗ Enter a number.")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "─"*58)
    print("  INPUT SUMMARY")
    print("─"*58)
    print(f"  Radars    : {N}  at positions {radars}")
    print(f"  Radius    : R = {R}")
    print(f"  Boundary  : K = {K} points")
    print(f"  Time slots: M = {M}")
    print(f"  Cooling   : L = {L},  C = {C_cool}  (window = {L+C_cool})")
    print(f"  Budgets   : B = {B}")
    print(f"  Time limit: {tl}s per phase")
    print("─"*58)

    confirm = input("\n  Proceed with these inputs? (y/n): ").strip().lower()
    if confirm != 'y':
        print("  Restarting input...\n")
        return collect_inputs()   # restart from top

    return dict(radars=radars, K=K, R=R, M=M, L=L, C_cool=C_cool,
                B=B, time_limit=tl)


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    params = collect_inputs()
    run(**params)
