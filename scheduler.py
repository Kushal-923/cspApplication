"""
scheduler.py  —  Two-Phase CSP / ILP Radar Scheduler with Frequency Assignment
================================================================================

Changes over scheduler_1_2 (1).py (now renamed to scheduler.py):

  [FUNCTION 2] Frequency-band model
    Each radar type now carries a "frequency_bands" list in
    radar_type_config.json.  assign_frequency_bands() reads this and returns,
    per radar, the list of band names it may operate on.  If a type has no
    "frequency_bands" key (old config), a single synthetic band ["DEFAULT"]
    is used so the pipeline still works — the scheduler just adds one trivial
    frequency variable that is always forced to DEFAULT when ON.

  [FUNCTION 3] Frequency assignment variables F[i][t][b]
    Every solver phase now declares:
        F[i][t][b_idx]  -- BoolVar: radar i uses band b_idx at time t
    with the coupling constraint:
        sum_b F[i][t][b] == X[i][t]   for all i, t
    Meaning:
        X[i][t]=0  →  all F[i][t][b]=0  (no band assigned while OFF)
        X[i][t]=1  →  exactly one F[i][t][b]=1  (exactly one band while ON)
    Only bands in the radar's supported list have variables — it is
    structurally impossible to assign an unsupported band.

  [FUNCTION 4] Frequency hopping constraints (configurable)
    Four optional constraint groups, all controlled by config constants below:
      A. FREQUENCY_HOPPING_ENABLED — if False, band must remain the same
         across consecutive ON slots (an OFF slot between them resets
         the obligation, since there is no active frequency during OFF).
      B. MINIMUM_DWELL_SLOTS (D_MIN) — if a radar switches to band b at
         slot t and stays ON at slots t+1..t+D_MIN-1, those ON slots must
         also use band b.  OFF slots in that window break the window; the
         dwell counter resets at the next ON slot.
      C. MAXIMUM_DWELL_SLOTS (D_MAX>0) — in any window of D_MAX+1
         consecutive slots, the radar may not use the same band for all
         D_MAX+1 of those slots (counting only ON slots, since F[i][t][b]=0
         when X[i][t]=0).
      D. FREQUENCY_TRANSITIONS (dict or None) — if provided, maps each band
         name to a list of bands it may transition to.  If None, any
         supported band may follow any other.

  Backward compatibility:
    - "schedule" key in schedule_output.json is unchanged.
    - New keys "frequency_schedule" and "frequency_bands_used" are added.
      generate_czml.py only reads "schedule" → no changes required there.
    - If radar_type_config.json has no "frequency_bands" for a type,
      a default ["DEFAULT"] band is synthesized — old configs still work.

Previous changes (v1.1 → v1.2, now v1.3):
  v1.2: BUDGETS and cooling (L, C) loaded per-radar-type from
        radar_type_config.json instead of hardcoded.
  v1.3: Frequency assignment and hopping (this file).

Input files (all REQUIRED):
  c_matrix.json          — built by build_matrix_1.3.py
  radar_meta.json        — built by build_matrix_1.3.py
  radar_type_config.json — built by optimize_placement.py (per-type
                            energy_budget/cooling_L/cooling_C/frequency_bands)
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

# ── Frequency hopping configuration ──────────────────────────────────────────
#
# FREQUENCY_HOPPING_ENABLED
#   True  : the scheduler may change frequency between consecutive ON slots,
#           subject to the dwell constraints below.
#   False : once a radar is ON, its frequency must not change until it goes
#           OFF.  An OFF slot between two ON slots resets this obligation —
#           the radar may resume on any supported band after the OFF gap.
#
FREQUENCY_HOPPING_ENABLED = True

# MINIMUM_DWELL_SLOTS (D_MIN)
#   Minimum number of consecutive active (ON) slots a radar must stay on the
#   same frequency before it may change.  1 = no minimum (can change every
#   slot).  Only active slots count toward dwell; an OFF slot resets the
#   dwell counter so the next ON slot is free to pick any band.
#
#   Example (D_MIN=2):
#     A A B B C C   — valid   (2 slots on A, then 2 on B, then 2 on C)
#     A B A B       — invalid (only 1 slot on A before switching to B)
#
MINIMUM_DWELL_SLOTS = 1   # default: no minimum (hopping each slot allowed)

# MAXIMUM_DWELL_SLOTS (D_MAX)
#   Maximum number of consecutive active slots a radar may use the SAME
#   band.  0 = no maximum (unlimited consecutive slots on one band).
#   Only ON slots count; OFF slots do not consume dwell budget.
#
#   Example (D_MAX=3):
#     A A A B       — valid   (3 consecutive A slots, then changes)
#     A A A A       — invalid (4 consecutive A slots)
#
MAXIMUM_DWELL_SLOTS = 0   # default: no maximum

# FREQUENCY_TRANSITIONS
#   Optional dict mapping each band name to the list of bands it may
#   transition to.  None = any supported band may follow any other band
#   (unrestricted hopping subject to dwell constraints above).
#
#   Example:
#     FREQUENCY_TRANSITIONS = {
#         "VHF":    ["UHF"],
#         "UHF":    ["VHF", "L_BAND"],
#         "L_BAND": ["UHF", "S_BAND"],
#         "S_BAND": ["L_BAND"],
#     }
#
#   Note: only bands listed in a type's frequency_bands need entries here.
#   Bands not present as keys are treated as "may transition to any band".
#
FREQUENCY_TRANSITIONS = None   # default: no restrictions


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
    cooling_L/cooling_C/frequency_bands.  energy_budget/cooling_L/cooling_C
    and frequency_bands are consumed here.
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


def assign_frequency_bands(meta, type_config):
    """
    [FUNCTION 2] Derive per-radar supported frequency bands from
    radar_type_config.json.

    For each radar, looks up its type's "frequency_bands" list in type_config.
    If the key is absent (old config without frequency_bands), defaults to
    ["DEFAULT"] with a printed warning — this keeps old configs working while
    making it clear the feature is not fully configured.

    Validates that:
      - frequency_bands is a non-empty list of strings
      - No duplicate band names within a type

    Returns a list of lists:
        supported_bands[i] = ["VHF", "UHF", ...]  for radar i
    """
    supported_bands = []
    warned_types = set()

    for r in meta:
        rtype = r["type"]
        cfg = type_config.get(rtype, {})

        if "frequency_bands" not in cfg:
            if rtype not in warned_types:
                print(
                    f"    WARNING: type '{rtype}' has no 'frequency_bands' in "
                    f"{TYPE_CONFIG_FILE}. Defaulting to [\"DEFAULT\"]. "
                    "Add 'frequency_bands' to the type config to enable proper "
                    "frequency assignment."
                )
                warned_types.add(rtype)
            supported_bands.append(["DEFAULT"])
            continue

        bands = cfg["frequency_bands"]

        # Validate: must be a non-empty list of strings with no duplicates
        if not isinstance(bands, list) or len(bands) == 0:
            raise RuntimeError(
                f"Type '{rtype}': 'frequency_bands' must be a non-empty list of "
                f"strings (got {bands!r}). Check {TYPE_CONFIG_FILE}."
            )
        for b in bands:
            if not isinstance(b, str):
                raise RuntimeError(
                    f"Type '{rtype}': all frequency band names must be strings "
                    f"(got {b!r} in {bands!r}). Check {TYPE_CONFIG_FILE}."
                )
        if len(bands) != len(set(bands)):
            dupes = sorted({b for b in bands if bands.count(b) > 1})
            raise RuntimeError(
                f"Type '{rtype}': duplicate band name(s) {dupes} in "
                f"'frequency_bands'. Each band must appear at most once."
            )

        supported_bands.append(list(bands))

    return supported_bands


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


def build_frequency_vars(model, N, M, supported_bands):
    """
    [FUNCTION 3] Build frequency assignment BoolVars F[i][t][b_idx].

    F[i][t][b_idx] = 1 means radar i is using its (b_idx)-th supported band
    at time slot t.  Variables only exist for bands in the radar's supported
    list — it is structurally impossible to assign an unsupported band.

    Returns:
        F : list of shape [N][M][len(supported_bands[i])] of BoolVar
    """
    F = []
    for i in range(N):
        n_bands = len(supported_bands[i])
        F_i = []
        for t in range(M):
            F_it = [
                model.NewBoolVar(f"f_{i}_{t}_{b}")
                for b in range(n_bands)
            ]
            F_i.append(F_it)
        F.append(F_i)
    return F


def add_frequency_constraints(model, X, F, N, M, supported_bands):
    """
    [FUNCTION 3] Core frequency-assignment constraint:
        sum_b F[i][t][b] == X[i][t]   for all i, t

    This encodes:
      - X[i][t]=0  →  all F[i][t][b]=0  (no band assigned when OFF)
      - X[i][t]=1  →  exactly one F[i][t][b]=1  (exactly one band when ON)

    [FUNCTION 4] Frequency hopping constraints (applied on top):
      A. If hopping is disabled: frequency must be the same across consecutive
         ON slots.
      B. Minimum dwell: once band b is chosen at ON slot t, all ON slots
         t+1..t+D_MIN-1 must also use band b.
      C. Maximum dwell: in any D_MAX+1 consecutive slots, no band may appear
         in all D_MAX+1 of those slots.
      D. Valid transitions: if FREQUENCY_TRANSITIONS is set, only listed
         band→band moves are allowed across consecutive ON slots.
    """
    D_MIN = MINIMUM_DWELL_SLOTS
    D_MAX = MAXIMUM_DWELL_SLOTS
    hop_enabled = FREQUENCY_HOPPING_ENABLED
    trans = FREQUENCY_TRANSITIONS  # dict or None

    for i in range(N):
        bands = supported_bands[i]
        n_bands = len(bands)
        band_idx = {b: idx for idx, b in enumerate(bands)}

        for t in range(M):
            # ── Core: sum_b F[i][t][b] == X[i][t] ──────────────────────────
            model.Add(sum(F[i][t]) == X[i][t])

        # ── A. Hopping disabled: same frequency across consecutive ON slots ──
        # If both slot t and slot t+1 are ON, they must use the same band.
        # Implementation: for each band b and each consecutive pair (t, t+1),
        #   F[i][t][b] + X[i][t] + X[i][t+1] <= F[i][t+1][b] + 2
        # which rearranges to:
        #   F[i][t+1][b] >= F[i][t][b] + X[i][t] + X[i][t+1] - 2
        # meaning: if t and t+1 are both ON and t is on band b, then t+1 must
        # also be on band b.
        if not hop_enabled:
            for t in range(M - 1):
                for b_idx in range(n_bands):
                    model.Add(
                        F[i][t + 1][b_idx] >= F[i][t][b_idx] + X[i][t] + X[i][t + 1] - 2
                    )

        # ── B. Minimum dwell: stay on same band for D_MIN active slots ──────
        # The dwell obligation starts when a radar TRANSITIONS to a new band.
        # Specifically: if at slot t the radar uses band b (F[i][t][b]=1) AND
        # at slot t-1 the radar was NOT on band b (either OFF or on a different
        # band), then the "start of a dwell window" has occurred. For each
        # subsequent slot t+k (k=1..D_MIN-1) where the radar is also ON, it
        # must also be on band b.
        #
        # We detect a "transition to band b at slot t" via a new BoolVar:
        #   start_b[i][t][b] = 1 iff:
        #     - slot t:   radar uses band b (F[i][t][b]=1)
        #     - slot t-1: radar was NOT on band b (F[i][t-1][b]=0 OR X[i][t-1]=0)
        #
        # Rather than creating explicit start variables (which adds O(N*M*B)
        # extra variables), we use an implication chain:
        #
        # For each (i, t, b_idx), if the radar was already on band b at t-1
        # (i.e., F[i][t-1][b]=1 AND X[i][t-1]=1), the dwell window started
        # at t-1 or earlier — no NEW window starts at t.  We only need the
        # forward dwell constraint to fire at true start-of-dwell points.
        #
        # The cleanest CP-SAT formulation without auxiliary variables:
        # For each (i, t, b, k) with k in 1..D_MIN-1 and t+k < M:
        #   If the radar switched TO band b exactly at slot t (meaning
        #   it's on b at t but NOT on b at t-1), AND it's ON at t+k, then
        #   it must be on b at t+k.
        #
        # "Switched to b at t" ≡ F[i][t][b]=1 AND (X[i][t-1]=0 OR F[i][t-1][b]=0)
        # Equivalent: F[i][t][b]=1 AND F[i][t-1][b] + (1 - X[i][t-1]) >= 1
        #           = F[i][t][b] - F[i][t-1][b] - (X[i][t-1]-1) >= 1
        # We write this as a combined linear constraint.
        #
        # Practical CP-SAT encoding (no auxiliary variable needed):
        # For t >= 1:
        #   F[i][t+k][b] >= F[i][t][b] - F[i][t-1][b] + X[i][t+k] - 1
        # Because:
        #   - If F[i][t][b]=0: RHS <= X[i][t+k]-1 <= 0, trivially satisfied
        #   - If F[i][t][b]=1 AND F[i][t-1][b]=1: RHS = X[i][t+k]-1 <= 0, trivially satisfied
        #     (dwell started earlier, no new obligation here)
        #   - If F[i][t][b]=1 AND F[i][t-1][b]=0: RHS = X[i][t+k], so
        #     F[i][t+k][b] >= X[i][t+k] → if ON at t+k, must use b (correct!)
        #
        # At t=0 (no previous slot), the dwell obligation always applies when
        # F[i][0][b]=1 (any band chosen at the first active slot starts dwell):
        #   F[i][k][b] >= F[i][0][b] + X[i][k] - 1   for k in 1..D_MIN-1
        #
        if hop_enabled and D_MIN > 1:
            for t in range(M):
                for k in range(1, min(D_MIN, M - t)):
                    for b_idx in range(n_bands):
                        if t == 0:
                            # First slot: dwell window always starts here
                            model.Add(
                                F[i][t + k][b_idx]
                                >= F[i][t][b_idx] + X[i][t + k] - 1
                            )
                        else:
                            # Dwell window starts only on a transition to b_idx
                            # F[t][b] - F[t-1][b] detects "newly switched to b"
                            model.Add(
                                F[i][t + k][b_idx]
                                >= F[i][t][b_idx]
                                - F[i][t - 1][b_idx]
                                + X[i][t + k]
                                - 1
                            )

        # ── C. Maximum dwell: cannot stay on same band > D_MAX active slots ─
        # In any window of D_MAX+1 consecutive slots, the sum of F[i][t][b]
        # across those slots must be <= D_MAX.  Since F[i][t][b]=0 when
        # X[i][t]=0 (enforced by the sum==X constraint above), OFF slots
        # contribute 0 and do not inflate the count — the constraint correctly
        # limits only active-slot dwell.
        if D_MAX > 0:
            for t in range(M - D_MAX):
                for b_idx in range(n_bands):
                    model.Add(
                        sum(F[i][t + k][b_idx] for k in range(D_MAX + 1)) <= D_MAX
                    )

        # ── D. Valid frequency transitions ───────────────────────────────────
        # If FREQUENCY_TRANSITIONS is provided, for each consecutive ON pair
        # (t, t+1), the transition from band b_from to band b_to must be in
        # the allowed list.  Implementation: for each (b_from, b_to) pair
        # where b_to is NOT allowed after b_from:
        #   F[i][t][b_from] + F[i][t+1][b_to] + X[i][t] + X[i][t+1] <= 3
        # This forbids F[i][t][b_from]=1 AND F[i][t+1][b_to]=1 AND
        # X[i][t]=1 AND X[i][t+1]=1 simultaneously (would require sum=4>3).
        if hop_enabled and trans is not None:
            for t in range(M - 1):
                for b_from_idx, b_from in enumerate(bands):
                    if b_from not in trans:
                        continue  # no restriction listed for this band
                    allowed_to = set(trans[b_from])
                    for b_to_idx, b_to in enumerate(bands):
                        if b_to not in allowed_to:
                            # Transition b_from → b_to is forbidden between
                            # two consecutive ON slots.
                            model.Add(
                                F[i][t][b_from_idx]
                                + F[i][t + 1][b_to_idx]
                                + X[i][t]
                                + X[i][t + 1]
                                <= 3
                            )


def extract_frequency_schedule(solver, F, N, M, supported_bands):
    """
    Extract the frequency assignment from a solved model.

    Returns:
        freq_sched : list[list[str|None]]
            freq_sched[i][t] = band name if radar i is ON at slot t, else None
    """
    freq_sched = []
    for i in range(N):
        row = []
        for t in range(M):
            assigned = None
            for b_idx, band_name in enumerate(supported_bands[i]):
                if solver.Value(F[i][t][b_idx]) == 1:
                    assigned = band_name
                    break
            row.append(assigned)
        freq_sched.append(row)
    return freq_sched


def print_schedule(sched, N, M, B, meta, freq_sched=None):
    """Print ON/OFF schedule with optional frequency assignments.
    Full grid for M<=24, summary bars for larger M."""
    print("\n  FINAL SCHEDULE")
    if M <= 24:
        header = "                    " + " ".join(f"{t:2d}" for t in range(M))
        print(header)
        for i in range(N):
            on  = sum(sched[i])
            name = meta[i]["name"] if meta else f"R{i+1:02d}"
            row  = "  ".join(str(sched[i][t]) for t in range(M))
            print(f"  {name:14s} [{on:2d}/{B[i]}]  {row}")
        if freq_sched is not None:
            print("\n  FREQUENCY ASSIGNMENT (band at each ON slot, — = OFF)")
            for i in range(N):
                name = meta[i]["name"] if meta else f"R{i+1:02d}"
                freq_row = "  ".join(
                    f"{(freq_sched[i][t] or '—'):>7s}" for t in range(M)
                )
                print(f"  {name:14s}  {freq_row}")
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

def step0_feasibility(C_mat, N, K, M, L_list, C_list, B, supported_bands, time_limit):
    banner("STEP 0 — Feasibility Check (pure CSP)",
           "Can every point be covered at every slot?")

    model = cp_model.CpModel()
    X = [[model.NewBoolVar(f"x_{i}_{t}") for t in range(M)]
         for i in range(N)]

    for j in range(K):
        for t in range(M):
            model.Add(sum(C_mat[i][j] * X[i][t] for i in range(N)) >= 1)

    add_cooling_and_energy(model, X, N, M, L_list, C_list, B)

    # [FUNCTION 3] Add frequency variables and constraints
    F = build_frequency_vars(model, N, M, supported_bands)
    add_frequency_constraints(model, X, F, N, M, supported_bands)

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

def phase1_min_void(C_mat, N, K, M, L_list, C_list, B, supported_bands, time_limit):
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

    # [FUNCTION 3] Add frequency variables and constraints
    F = build_frequency_vars(model, N, M, supported_bands)
    add_frequency_constraints(model, X, F, N, M, supported_bands)

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

    # [FUNCTION 3] Extract frequency assignment
    freq_sched = extract_frequency_schedule(solver, F, N, M, supported_bands)

    print(f"\n  Solver status   : {label}")
    print(f"  Total void      : {total_void}  space-time cells uncovered")
    print(f"  Void density    : {void_pct:.2f}%  of ({K} pts × {M} slots)")
    print(f"  Total ON-slots  : {total_on}")

    return {
        "phase":              1,
        "status":             label,
        "total_void":         total_void,
        "void_pct":           void_pct,
        "total_on":           total_on,
        "schedule":           sched,
        "void_matrix":        void_mat,
        # [FUNCTION 3] Frequency output (backward-compatible new keys)
        "frequency_schedule": freq_sched,
        "frequency_bands_used": sorted(
            {b for bands in supported_bands for b in bands}
        ),
    }


# ══════════════════════════════════════════════════════════════
#  PHASE 2 — MINIMISE OVERLAP
# ══════════════════════════════════════════════════════════════

def phase2_min_overlap(C_mat, N, K, M, L_list, C_list, B, supported_bands, time_limit):
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

    # [FUNCTION 3] Add frequency variables and constraints
    F = build_frequency_vars(model, N, M, supported_bands)
    add_frequency_constraints(model, X, F, N, M, supported_bands)

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

    # [FUNCTION 3] Extract frequency assignment
    freq_sched = extract_frequency_schedule(solver, F, N, M, supported_bands)

    print(f"\n  Solver status   : {label}")
    print(f"  Total overlap   : {total_ov}  extra radar-point coverages")
    print(f"  Overlap density : {ov_den:.4f}  extra radars per (point × slot)")
    print(f"  Total ON-slots  : {total_on}")
    print(f"  Void            : 0  (hard constraint enforced)")

    return {
        "phase":              2,
        "status":             label,
        "total_overlap":      total_ov,
        "overlap_density":    ov_den,
        "total_void":         0,
        "total_on":           total_on,
        "schedule":           sched,
        "overlap_matrix":     over_mat,
        # [FUNCTION 3] Frequency output (backward-compatible new keys)
        "frequency_schedule": freq_sched,
        "frequency_bands_used": sorted(
            {b for bands in supported_bands for b in bands}
        ),
    }


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "═" * 60)
    print("  SCHEDULER v1.3  —  Two-Phase Radar Scheduler + Frequency")
    print("═" * 60)

    # ── Load matrix ───────────────────────────────────────────
    print(f"\n[1] Loading {MATRIX_FILE} ...")
    with open(MATRIX_FILE) as f:
        C_mat = json.load(f)

    N = len(C_mat)
    K = len(C_mat[0])
    M = NUM_TIME_SLOTS

    # ── Load metadata ─────────────────────────────────────────
    try:
        with open(META_FILE) as f:
            meta = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(
            f"{META_FILE} not found. This scheduler now requires {META_FILE} "
            f"(for radar type lookup) and {TYPE_CONFIG_FILE} (for per-type "
            "energy_budget/cooling_L/cooling_C/frequency_bands) -- there is "
            "no more hardcoded fallback. Run build_matrix_1.3.py first to "
            "produce it."
        )
    print(f"    Loaded radar metadata: {len(meta)} radars")

    # ── Load per-type config ──────────────────────────────────
    try:
        type_config = load_type_config(TYPE_CONFIG_FILE)
    except FileNotFoundError:
        raise RuntimeError(
            f"{TYPE_CONFIG_FILE} not found. This scheduler now requires "
            f"{TYPE_CONFIG_FILE} for per-type energy_budget/cooling_L/"
            "cooling_C/frequency_bands -- there is no more hardcoded "
            "fallback. Run optimize_placement.py first to produce it."
        )
    print(f"    Loaded config for {len(type_config)} type(s): {sorted(type_config.keys())}")

    # ── Derive budgets + cooling ───────────────────────────────
    BUDGETS, L_LIST, C_LIST = assign_budgets_and_cooling(meta, type_config, M)

    # ── [FUNCTION 2] Derive per-radar supported frequency bands ──
    SUPPORTED_BANDS = assign_frequency_bands(meta, type_config)

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

    # ── [FUNCTION 2] Print frequency band summary ─────────────
    print(f"\n  Frequency hopping : {'ENABLED' if FREQUENCY_HOPPING_ENABLED else 'DISABLED'}")
    print(f"  Min dwell slots   : {MINIMUM_DWELL_SLOTS}  (1=no minimum)")
    print(f"  Max dwell slots   : {MAXIMUM_DWELL_SLOTS}  (0=no maximum)")
    print(f"  Transition graph  : {'CUSTOM' if FREQUENCY_TRANSITIONS else 'UNRESTRICTED'}")
    print(f"\n  Frequency bands per radar type:")
    for rtype, cfg in sorted(type_config.items()):
        bands = cfg.get("frequency_bands", ["DEFAULT (fallback)"])
        print(f"    {rtype:12s} → {bands}")
    # Show per-radar band assignments (only first 10 to avoid clutter)
    print(f"\n  Per-radar supported bands (first {min(10, N)}):")
    for i in range(min(10, N)):
        name = meta[i]["name"]
        print(f"    {name:14s} → {SUPPORTED_BANDS[i]}")
    if N > 10:
        print(f"    ... ({N - 10} more radars)")

    # Validate budget/cooling lists
    if len(BUDGETS) != N or len(L_LIST) != N or len(C_LIST) != N:
        raise ValueError(
            f"BUDGETS/L_LIST/C_LIST have {len(BUDGETS)}/{len(L_LIST)}/{len(C_LIST)} "
            f"entries but N={N} radars. Check assign_budgets_and_cooling() or "
            f"{TYPE_CONFIG_FILE}."
        )
    if len(SUPPORTED_BANDS) != N:
        raise ValueError(
            f"SUPPORTED_BANDS has {len(SUPPORTED_BANDS)} entries but N={N} radars. "
            "Check assign_frequency_bands()."
        )

    # ── Coverage stats ────────────────────────────────────────
    uncovered_pts = print_coverage_stats(C_mat, N, K)

    if uncovered_pts == K:
        raise RuntimeError(
            "Every boundary point is uncoverable — check radar ranges and positions."
        )

    # ── Step 0 ────────────────────────────────────────────────
    result = step0_feasibility(
        C_mat, N, K, M, L_LIST, C_LIST, BUDGETS, SUPPORTED_BANDS, TIME_LIMIT
    )

    # ── Branch ────────────────────────────────────────────────
    if result == "FEASIBLE":
        output = phase2_min_overlap(
            C_mat, N, K, M, L_LIST, C_LIST, BUDGETS, SUPPORTED_BANDS, TIME_LIMIT
        )
    else:
        output = phase1_min_void(
            C_mat, N, K, M, L_LIST, C_LIST, BUDGETS, SUPPORTED_BANDS, TIME_LIMIT
        )

    # ── Print schedule ────────────────────────────────────────
    if "schedule" in output:
        freq_sched = output.get("frequency_schedule")
        print_schedule(output["schedule"], N, M, BUDGETS, meta, freq_sched)

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

    # Print frequency summary
    if "frequency_bands_used" in output:
        bands_used = output["frequency_bands_used"]
        print(f"  Bands in use    : {bands_used}")
        if "frequency_schedule" in output and output["frequency_schedule"]:
            freq_sched = output["frequency_schedule"]
            # Count assignments per band
            band_counts = {}
            for i in range(N):
                for t in range(M):
                    b = freq_sched[i][t]
                    if b is not None:
                        band_counts[b] = band_counts.get(b, 0) + 1
            print(f"  Freq assignment : " + ", ".join(
                f"{b}={band_counts.get(b, 0)}" for b in bands_used
            ))
    print()

    with open("schedule_output.json", "w") as f:
        json.dump(output, f)
    print(f"  Saved schedule_output.json  (keys: schedule, frequency_schedule, "
          f"frequency_bands_used + existing keys)")
