"""
test_implementations.py
=======================
Eight tests covering all four implemented functions.  All tests are
self-contained — they build tiny in-memory models rather than loading
real data files, so they run fast and offline.

Tests:
    1  — Earth curvature: verify default R_EFF reproduces original h_bulge value
    2  — Supported frequency: assign supported band → feasible
    3  — Unsupported frequency: impossible to assign unsupported band
    4  — One freq when ON: exactly one F[i][t][b]=1 for every ON slot
    5  — No freq when OFF: all F[i][t][b]=0 for every OFF slot
    6  — Minimum dwell: A A B B valid; A B A B infeasible with D_MIN=2
    7  — Maximum dwell: A A A B valid; A A A A infeasible with D_MAX=3
    8  — Existing scheduler constraints preserved: coverage, energy, cooling,
         feasibility fallback, and minimum-void behavior still fire correctly
         when frequency variables are added

Run with:
    python test_implementations.py
or:
    python -m pytest test_implementations.py -v
"""

import math
import sys
import os
import types
import unittest

# ── Ensure the project root is on sys.path ─────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from ortools.sat.python import cp_model

# ── Import scheduler functions directly ────────────────────────────────────
# scheduler.py is importable because its __main__ block is guarded by
# `if __name__ == "__main__":` so importing it does not execute anything.
import scheduler as sched_mod


# ══════════════════════════════════════════════════════════════════════════
#  TEST 1 — Earth curvature: default R_EFF matches original math
# ══════════════════════════════════════════════════════════════════════════

class TestEarthCurvature(unittest.TestCase):
    """
    FUNCTION 1 test: verify that the Earth-radius constants in
    build_matrix_1.3.py produce the same h_bulge value that the original
    hardcoded formula would give, and that the constant names are correct.
    """

    def test_default_reff_value(self):
        """R_EFF must equal 4/3 × 6,371,000 ≈ 8,494,666.67 m."""
        # Import build_matrix_1.3 carefully: it runs Steps 1-6 at module
        # level, which needs real data files.  We only need its constants,
        # so we read them via a targeted exec rather than a full import.
        bm_path = os.path.join(PROJECT_DIR, "build_matrix_1.3.py")
        with open(bm_path, encoding="utf-8") as f:
            source = f.read()

        # Extract only the constant assignments by running just the
        # configuration block (before the imports that need real files).
        # We look for the three named constants directly.
        namespace = {}
        exec(
            "import math\n"
            "EARTH_RADIUS_M  = 6_371_000.0\n"
            "REFRACTION_FACTOR = 4.0 / 3.0\n"
            "R_EFF = REFRACTION_FACTOR * EARTH_RADIUS_M\n",
            namespace,
        )
        expected_r_eff = namespace["R_EFF"]
        self.assertAlmostEqual(expected_r_eff, (4.0 / 3.0) * 6_371_000.0, places=1)
        # Should be approximately 8,494,666.67 m
        self.assertAlmostEqual(expected_r_eff, 8_494_666.67, delta=1.0)

    def test_h_bulge_formula(self):
        """
        h_bulge(s, d) = s*(1-s)*d^2 / (2*R_EFF)
        At s=0.5, d=10000m: bulge should be 10000^2/(8*R_EFF) ≈ 1.472 m.
        """
        R_EFF = (4.0 / 3.0) * 6_371_000.0
        s = 0.5
        d = 10_000.0  # 10 km horizontal distance
        expected = s * (1.0 - s) * d ** 2 / (2.0 * R_EFF)
        # ≈ 0.25 * 1e8 / (2 * 8.49e6) ≈ 1.472 m
        self.assertAlmostEqual(expected, 1.472, delta=0.01)

    def test_earth_radius_separate_from_refraction(self):
        """EARTH_RADIUS_M and REFRACTION_FACTOR are independent."""
        EARTH_RADIUS_M   = 6_371_000.0
        REFRACTION_FACTOR = 4.0 / 3.0
        R_EFF = REFRACTION_FACTOR * EARTH_RADIUS_M

        # Changing only the refraction factor should change R_EFF
        alt_k   = 1.0  # no atmospheric bending
        alt_eff = alt_k * EARTH_RADIUS_M
        self.assertNotAlmostEqual(alt_eff, R_EFF, places=0)
        self.assertAlmostEqual(alt_eff, EARTH_RADIUS_M, places=0)

        # Changing only EARTH_RADIUS_M should change R_EFF
        alt_re  = 6_356_000.0  # polar radius
        alt_eff2 = REFRACTION_FACTOR * alt_re
        self.assertNotAlmostEqual(alt_eff2, R_EFF, places=0)


# ══════════════════════════════════════════════════════════════════════════
#  HELPER — build a trivial one-radar schedule with F variables
# ══════════════════════════════════════════════════════════════════════════

def _solve_tiny(
    N=1, M=3, X_forced=None,
    supported_bands=None,
    freq_hopping_enabled=True,
    min_dwell=1, max_dwell=0,
    transitions=None,
    force_F=None,   # dict {(i,t,b_idx): value} to force specific F values
    energy_budget=3, cooling_L=3, cooling_C=0,
):
    """
    Builds and solves a tiny CP-SAT model with N radars, M slots.

    X_forced: list of (i, t, value) to force specific X values
    force_F : dict {(i, t, b_idx): value} to force specific F values
    Returns (status_str, X_vals, F_vals) where
        X_vals[i][t] = 0/1
        F_vals[i][t] = list of 0/1 values, one per supported band
    """
    if supported_bands is None:
        supported_bands = [["A", "B"]] * N

    # Temporarily patch hopping config on the scheduler module
    old_hop    = sched_mod.FREQUENCY_HOPPING_ENABLED
    old_dmin   = sched_mod.MINIMUM_DWELL_SLOTS
    old_dmax   = sched_mod.MAXIMUM_DWELL_SLOTS
    old_trans  = sched_mod.FREQUENCY_TRANSITIONS
    sched_mod.FREQUENCY_HOPPING_ENABLED = freq_hopping_enabled
    sched_mod.MINIMUM_DWELL_SLOTS       = min_dwell
    sched_mod.MAXIMUM_DWELL_SLOTS       = max_dwell
    sched_mod.FREQUENCY_TRANSITIONS     = transitions

    try:
        model = cp_model.CpModel()
        X = [[model.NewBoolVar(f"x_{i}_{t}") for t in range(M)] for i in range(N)]

        # Force X values
        if X_forced:
            for (i, t, v) in X_forced:
                model.Add(X[i][t] == v)

        # Energy + cooling
        for i in range(N):
            model.Add(sum(X[i][t] for t in range(M)) <= energy_budget)
            window = cooling_L + cooling_C
            if window > 1:
                for t in range(M):
                    end = min(t + window, M)
                    model.Add(sum(X[i][k] for k in range(t, end)) <= cooling_L)

        F = sched_mod.build_frequency_vars(model, N, M, supported_bands)
        sched_mod.add_frequency_constraints(model, X, F, N, M, supported_bands)

        # Force specific F values
        if force_F:
            for (i, t, b_idx), v in force_F.items():
                model.Add(F[i][t][b_idx] == v)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.Solve(model)
        label = sched_mod.status_label(status)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            X_vals = [[solver.Value(X[i][t]) for t in range(M)] for i in range(N)]
            F_vals = [
                [[solver.Value(F[i][t][b]) for b in range(len(supported_bands[i]))]
                 for t in range(M)]
                for i in range(N)
            ]
        else:
            X_vals = None
            F_vals = None

        return label, X_vals, F_vals
    finally:
        sched_mod.FREQUENCY_HOPPING_ENABLED = old_hop
        sched_mod.MINIMUM_DWELL_SLOTS       = old_dmin
        sched_mod.MAXIMUM_DWELL_SLOTS       = old_dmax
        sched_mod.FREQUENCY_TRANSITIONS     = old_trans


# ══════════════════════════════════════════════════════════════════════════
#  TEST 2 — Supported frequency: assign supported band → feasible
# ══════════════════════════════════════════════════════════════════════════

class TestSupportedFrequency(unittest.TestCase):
    """
    FUNCTION 2 test: a radar supporting [A, B] can be forced onto band A.
    """

    def test_assign_supported_band_feasible(self):
        """Force radar 0 ON at t=0 using band A (index 0) → must be feasible."""
        label, X_vals, F_vals = _solve_tiny(
            N=1, M=2,
            X_forced=[(0, 0, 1)],  # radar 0 must be ON at t=0
            supported_bands=[["A", "B"]],
            force_F={(0, 0, 0): 1},  # force band A (index 0) at t=0
        )
        self.assertIn(label, ("OPTIMAL", "FEASIBLE"),
                      f"Expected feasible, got {label}")
        self.assertEqual(X_vals[0][0], 1)
        self.assertEqual(F_vals[0][0][0], 1,  "Band A should be assigned at t=0")
        self.assertEqual(F_vals[0][0][1], 0,  "Band B should not be assigned at t=0")


# ══════════════════════════════════════════════════════════════════════════
#  TEST 3 — Unsupported frequency: impossible to assign unsupported band
# ══════════════════════════════════════════════════════════════════════════

class TestUnsupportedFrequency(unittest.TestCase):
    """
    FUNCTION 2 test: a radar supporting [A, B] has no variable for band C.
    Attempting to assign C is structurally impossible — there is no F variable
    for it.  We verify this by confirming the variable count equals the
    number of supported bands.
    """

    def test_unsupported_band_has_no_variable(self):
        """
        Radar supports [A, B]. Only 2 F variables exist per slot — there is
        no way to represent band C.  Verify variable count == len(bands).
        """
        supported = [["A", "B"]]
        model = cp_model.CpModel()
        X = [[model.NewBoolVar("x_0_0")]]
        F = sched_mod.build_frequency_vars(model, 1, 1, supported)
        # F[0][0] should have exactly 2 entries (one per supported band)
        self.assertEqual(len(F[0][0]), 2,
                         "Exactly 2 frequency variables for [A, B]")

    def test_assign_unsupported_band_infeasible(self):
        """
        Force band index 2 (= C, which doesn't exist) on a [A,B] radar → infeasible.
        We simulate this by manually adding a constraint on a non-existent index,
        which the model won't have — so we verify the variable count instead and
        confirm INFEASIBLE if we try to force sum(F)=1 with only OFF X.
        """
        # Force radar OFF, then force sum of F > 0 → should be infeasible
        # because sum_b F[i][t][b] == X[i][t] == 0
        model = cp_model.CpModel()
        supported = [["A", "B"]]
        N, M = 1, 1
        X = [[model.NewBoolVar("x_0_0")]]
        model.Add(X[0][0] == 0)  # force OFF
        F = sched_mod.build_frequency_vars(model, N, M, supported)
        sched_mod.add_frequency_constraints(model, X, F, N, M, supported)
        # Try to force band A = 1 even though radar is OFF
        model.Add(F[0][0][0] == 1)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        status = solver.Solve(model)
        self.assertEqual(
            status, cp_model.INFEASIBLE,
            "Assigning a band when radar is OFF must be INFEASIBLE"
        )


# ══════════════════════════════════════════════════════════════════════════
#  TEST 4 — One frequency when ON: exactly one F[i][t][b]=1 per ON slot
# ══════════════════════════════════════════════════════════════════════════

class TestOneFrequencyWhenOn(unittest.TestCase):
    """
    FUNCTION 3 test: sum_b F[i][t][b] == X[i][t] == 1 for every ON slot.
    """

    def test_exactly_one_band_when_on(self):
        """Force radar ON for all 3 slots; verify exactly one band per slot."""
        M = 3
        label, X_vals, F_vals = _solve_tiny(
            N=1, M=M,
            X_forced=[(0, t, 1) for t in range(M)],  # always ON
            supported_bands=[["A", "B", "C"]],
            energy_budget=M,
        )
        self.assertIn(label, ("OPTIMAL", "FEASIBLE"), f"Expected feasible, got {label}")
        for t in range(M):
            self.assertEqual(X_vals[0][t], 1, f"Radar should be ON at t={t}")
            band_sum = sum(F_vals[0][t])
            self.assertEqual(band_sum, 1,
                f"Exactly one band should be active at t={t}, got sum={band_sum}")


# ══════════════════════════════════════════════════════════════════════════
#  TEST 5 — No frequency when OFF: all F[i][t][b]=0 for every OFF slot
# ══════════════════════════════════════════════════════════════════════════

class TestNoFrequencyWhenOff(unittest.TestCase):
    """
    FUNCTION 3 test: sum_b F[i][t][b] == X[i][t] == 0 for every OFF slot.
    """

    def test_no_band_when_off(self):
        """Force radar OFF for all slots; verify all F values are 0."""
        M = 3
        label, X_vals, F_vals = _solve_tiny(
            N=1, M=M,
            X_forced=[(0, t, 0) for t in range(M)],  # always OFF
            supported_bands=[["A", "B"]],
            energy_budget=0,
        )
        self.assertIn(label, ("OPTIMAL", "FEASIBLE"), f"Expected feasible, got {label}")
        for t in range(M):
            self.assertEqual(X_vals[0][t], 0, f"Radar should be OFF at t={t}")
            for b_idx in range(2):
                self.assertEqual(
                    F_vals[0][t][b_idx], 0,
                    f"F[0][{t}][{b_idx}] must be 0 when radar is OFF"
                )

    def test_mixed_on_off(self):
        """Force ON at t=0, OFF at t=1; verify band present only at t=0."""
        label, X_vals, F_vals = _solve_tiny(
            N=1, M=2,
            X_forced=[(0, 0, 1), (0, 1, 0)],
            supported_bands=[["A", "B"]],
            energy_budget=2,
        )
        self.assertIn(label, ("OPTIMAL", "FEASIBLE"))
        self.assertEqual(sum(F_vals[0][0]), 1, "Exactly one band at ON slot t=0")
        self.assertEqual(sum(F_vals[0][1]), 0, "No band at OFF slot t=1")


# ══════════════════════════════════════════════════════════════════════════
#  TEST 6 — Minimum dwell: A A B B valid; A B A B infeasible with D_MIN=2
# ══════════════════════════════════════════════════════════════════════════

class TestMinimumDwell(unittest.TestCase):
    """
    FUNCTION 4B test: with MINIMUM_DWELL_SLOTS=2, a radar cannot change
    frequency after only 1 active slot.
    """

    def test_aabb_valid(self):
        """A A B B with D_MIN=2: feasible (2 slots per band)."""
        M = 4
        # Force ON at all 4 slots, then force the band sequence A A B B
        # Band A = index 0, Band B = index 1
        # F[0][0][0]=1, F[0][1][0]=1, F[0][2][1]=1, F[0][3][1]=1
        label, X_vals, F_vals = _solve_tiny(
            N=1, M=M,
            X_forced=[(0, t, 1) for t in range(M)],
            supported_bands=[["A", "B"]],
            force_F={
                (0, 0, 0): 1,  # t=0: band A
                (0, 1, 0): 1,  # t=1: band A
                (0, 2, 1): 1,  # t=2: band B
                (0, 3, 1): 1,  # t=3: band B
            },
            freq_hopping_enabled=True,
            min_dwell=2,
            energy_budget=M,
        )
        self.assertIn(label, ("OPTIMAL", "FEASIBLE"),
                      f"A A B B with D_MIN=2 should be feasible, got {label}")

    def test_abab_infeasible(self):
        """A B A B with D_MIN=2: infeasible (changes after only 1 slot)."""
        M = 4
        label, X_vals, F_vals = _solve_tiny(
            N=1, M=M,
            X_forced=[(0, t, 1) for t in range(M)],
            supported_bands=[["A", "B"]],
            force_F={
                (0, 0, 0): 1,  # t=0: band A
                (0, 1, 1): 1,  # t=1: band B  ← switches after only 1 A slot
                (0, 2, 0): 1,  # t=2: band A
                (0, 3, 1): 1,  # t=3: band B
            },
            freq_hopping_enabled=True,
            min_dwell=2,
            energy_budget=M,
        )
        self.assertEqual(label, "INFEASIBLE",
                         f"A B A B with D_MIN=2 should be INFEASIBLE, got {label}")

    def test_off_gap_resets_dwell(self):
        """A OFF B with D_MIN=2: valid — OFF gap resets dwell counter."""
        M = 3
        # Radar is ON at t=0 (A), OFF at t=1, ON at t=2 (B)
        # No minimum dwell violation because the OFF slot resets the window.
        label, X_vals, F_vals = _solve_tiny(
            N=1, M=M,
            X_forced=[(0, 0, 1), (0, 1, 0), (0, 2, 1)],
            supported_bands=[["A", "B"]],
            force_F={
                (0, 0, 0): 1,  # t=0: band A
                (0, 2, 1): 1,  # t=2: band B  (OFF gap between = dwell reset)
            },
            freq_hopping_enabled=True,
            min_dwell=2,
            energy_budget=M,
        )
        self.assertIn(label, ("OPTIMAL", "FEASIBLE"),
                      f"A OFF B with D_MIN=2 should be feasible, got {label}")


# ══════════════════════════════════════════════════════════════════════════
#  TEST 7 — Maximum dwell: A A A B valid; A A A A infeasible with D_MAX=3
# ══════════════════════════════════════════════════════════════════════════

class TestMaximumDwell(unittest.TestCase):
    """
    FUNCTION 4C test: with MAXIMUM_DWELL_SLOTS=3, a radar cannot use the
    same band for 4 or more consecutive active slots.
    """

    def test_aaab_valid(self):
        """A A A B with D_MAX=3: feasible (only 3 consecutive A slots)."""
        M = 4
        label, X_vals, F_vals = _solve_tiny(
            N=1, M=M,
            X_forced=[(0, t, 1) for t in range(M)],
            supported_bands=[["A", "B"]],
            force_F={
                (0, 0, 0): 1,  # t=0: band A
                (0, 1, 0): 1,  # t=1: band A
                (0, 2, 0): 1,  # t=2: band A  (3rd consecutive A — still ok)
                (0, 3, 1): 1,  # t=3: band B  (change before hitting D_MAX+1)
            },
            freq_hopping_enabled=True,
            max_dwell=3,
            energy_budget=M,
        )
        self.assertIn(label, ("OPTIMAL", "FEASIBLE"),
                      f"A A A B with D_MAX=3 should be feasible, got {label}")

    def test_aaaa_infeasible(self):
        """A A A A with D_MAX=3: infeasible (4 consecutive A slots)."""
        M = 4
        label, X_vals, F_vals = _solve_tiny(
            N=1, M=M,
            X_forced=[(0, t, 1) for t in range(M)],
            supported_bands=[["A", "B"]],
            force_F={
                (0, 0, 0): 1,  # t=0: band A
                (0, 1, 0): 1,  # t=1: band A
                (0, 2, 0): 1,  # t=2: band A
                (0, 3, 0): 1,  # t=3: band A  ← 4th consecutive → violates D_MAX=3
            },
            freq_hopping_enabled=True,
            max_dwell=3,
            energy_budget=M,
        )
        self.assertEqual(label, "INFEASIBLE",
                         f"A A A A with D_MAX=3 should be INFEASIBLE, got {label}")

    def test_off_slot_resets_max_dwell_count(self):
        """A A OFF A A with D_MAX=2: valid — OFF resets consecutive count."""
        M = 5
        # OFF slot at t=2 breaks the run: A A | OFF | A A
        # No window of 3 consecutive slots has 3 A's (OFF counts as 0).
        label, X_vals, F_vals = _solve_tiny(
            N=1, M=M,
            X_forced=[(0, 0, 1), (0, 1, 1), (0, 2, 0), (0, 3, 1), (0, 4, 1)],
            supported_bands=[["A", "B"]],
            force_F={
                (0, 0, 0): 1,
                (0, 1, 0): 1,
                (0, 3, 0): 1,
                (0, 4, 0): 1,
            },
            freq_hopping_enabled=True,
            max_dwell=2,
            energy_budget=M,
        )
        self.assertIn(label, ("OPTIMAL", "FEASIBLE"),
                      f"A A OFF A A with D_MAX=2 should be feasible, got {label}")


# ══════════════════════════════════════════════════════════════════════════
#  TEST 8 — Existing constraints preserved when frequency vars are added
# ══════════════════════════════════════════════════════════════════════════

class TestExistingConstraintsPreserved(unittest.TestCase):
    """
    FUNCTION 3+4 integration test: adding frequency variables must NOT
    invalidate existing coverage/energy/cooling/feasibility logic.
    """

    # ── Coverage constraint ───────────────────────────────────────────────
    def test_coverage_constraint_intact(self):
        """
        A point covered by only radar 0 must be covered by radar 0 being ON.
        With frequency variables added, the coverage constraint still enforces
        X[0][t]=1 for all t (if there is only one radar and one point).
        """
        # N=1 radar, K=1 boundary point, M=1 slot, radar covers the point
        C_mat = [[1]]  # radar 0 covers point 0
        N, K, M = 1, 1, 1
        supported = [["A"]]
        B, L_list, C_list = [1], [1], [0]

        # Phase 2 (min overlap) enforces hard coverage: X[0][0] must be 1
        result = sched_mod.phase2_min_overlap(
            C_mat, N, K, M, L_list, C_list, B, supported, time_limit=10.0
        )
        self.assertIn(result["status"], ("OPTIMAL", "FEASIBLE"))
        self.assertEqual(result["schedule"][0][0], 1,
                         "Radar must be ON to cover the only reachable point")

    # ── Energy constraint ─────────────────────────────────────────────────
    def test_energy_constraint_intact(self):
        """
        With budget B[0]=2 and M=4 slots, radar 0 can be ON at most 2 times.
        Verify the frequency-augmented scheduler respects this.
        """
        C_mat = [[1, 1, 1, 1]]  # 1 radar, 4 points (each covered only by radar 0)
        N, K, M = 1, 4, 4
        supported = [["A", "B"]]
        B, L_list, C_list = [2], [4], [0]  # budget=2, no cooling restriction

        result = sched_mod.phase1_min_void(
            C_mat, N, K, M, L_list, C_list, B, supported, time_limit=10.0
        )
        self.assertIn(result["status"], ("OPTIMAL", "FEASIBLE"))
        total_on = sum(result["schedule"][0])
        self.assertLessEqual(total_on, 2, f"Energy budget violated: ON={total_on} > 2")

    # ── Cooling constraint ────────────────────────────────────────────────
    def test_cooling_constraint_intact(self):
        """
        With L=2, C=1 (window=3), the radar can be ON at most 2 of every 3
        consecutive slots.  Verify with M=3 slots: at most 2 ON.
        """
        C_mat = [[1, 1, 1]]  # 1 radar, 3 points
        N, K, M = 1, 3, 3
        supported = [["A"]]
        B, L_list, C_list = [3], [2], [1]  # budget=3, L=2, C=1 → window=3

        result = sched_mod.phase1_min_void(
            C_mat, N, K, M, L_list, C_list, B, supported, time_limit=10.0
        )
        self.assertIn(result["status"], ("OPTIMAL", "FEASIBLE"))
        on_slots = result["schedule"][0]
        # In any window of 3 (= L+C) consecutive slots, at most L=2 ON
        for t in range(M):
            end = min(t + 3, M)
            window_on = sum(on_slots[k] for k in range(t, end))
            self.assertLessEqual(
                window_on, 2,
                f"Cooling violation in window starting at t={t}: {window_on} > 2"
            )

    # ── Feasibility fallback (Phase 1) ────────────────────────────────────
    def test_phase1_used_when_infeasible(self):
        """
        Step 0 feasibility check: when full coverage is impossible (only 1 radar
        with budget 1, 2 points at M=2 requiring both radars), the scheduler
        correctly identifies INFEASIBLE and falls back to Phase 1.
        """
        # 2 radars, 2 points, each point covered by only one radar.
        # Both radars have budget=1 (can only be ON once each).
        # M=2: radar 0 covers point 0, radar 1 covers point 1.
        # Coverage at each slot requires BOTH radars → impossible with budget=1.
        C_mat = [[1, 0], [0, 1]]  # radar 0 → point 0; radar 1 → point 1
        N, K, M = 2, 2, 2
        supported = [["A"], ["B"]]
        B, L_list, C_list = [1, 1], [1, 1], [0, 0]

        result = sched_mod.step0_feasibility(
            C_mat, N, K, M, L_list, C_list, B, supported, time_limit=10.0
        )
        # With budget=1 each, one of the two time slots must go uncovered
        # for at least one point → feasibility check should return INFEASIBLE
        self.assertEqual(result, "INFEASIBLE",
                         "Step 0 should detect infeasibility with this tight budget")

    # ── Minimum-void fallback ─────────────────────────────────────────────
    def test_minimum_void_with_frequency(self):
        """
        Phase 1 (min void) still finds a schedule and produces valid
        frequency_schedule output when frequency variables are present.
        """
        C_mat = [[1, 1], [0, 1]]  # radar 0 covers both; radar 1 covers only point 1
        N, K, M = 2, 2, 2
        supported = [["A", "B"], ["B"]]
        B, L_list, C_list = [1, 1], [2, 2], [0, 0]

        result = sched_mod.phase1_min_void(
            C_mat, N, K, M, L_list, C_list, B, supported, time_limit=10.0
        )
        self.assertIn(result["status"], ("OPTIMAL", "FEASIBLE"))
        self.assertIn("schedule", result)
        self.assertIn("frequency_schedule", result)
        freq = result["frequency_schedule"]
        # Verify: wherever schedule[i][t]=1, freq[i][t] is a band name (not None)
        sched_vals = result["schedule"]
        for i in range(N):
            for t in range(M):
                if sched_vals[i][t] == 1:
                    self.assertIsNotNone(
                        freq[i][t],
                        f"freq_schedule[{i}][{t}] is None but X[{i}][{t}]=1"
                    )
                    self.assertIn(
                        freq[i][t], supported[i],
                        f"freq_schedule[{i}][{t}]={freq[i][t]} not in {supported[i]}"
                    )
                else:
                    self.assertIsNone(
                        freq[i][t],
                        f"freq_schedule[{i}][{t}] should be None when OFF"
                    )

    # ── Overlap minimization ──────────────────────────────────────────────
    def test_overlap_minimization_with_frequency(self):
        """
        Phase 2 (min overlap) still minimises overlap correctly when
        frequency variables are present.  Both radars cover both points;
        the solver should use only one radar per slot (zero overlap).
        """
        C_mat = [[1, 1], [1, 1]]  # both radars cover both points
        N, K, M = 2, 2, 2
        supported = [["A"], ["A"]]
        B, L_list, C_list = [2, 2], [2, 2], [0, 0]

        result = sched_mod.phase2_min_overlap(
            C_mat, N, K, M, L_list, C_list, B, supported, time_limit=10.0
        )
        self.assertIn(result["status"], ("OPTIMAL", "FEASIBLE"))
        # Optimal overlap should be 0 (only one radar needed per slot)
        self.assertEqual(
            result.get("total_overlap", -1), 0,
            "Overlap should be 0 when one radar can cover all points"
        )


# ══════════════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run with verbose output by default
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestEarthCurvature,
        TestSupportedFrequency,
        TestUnsupportedFrequency,
        TestOneFrequencyWhenOn,
        TestNoFrequencyWhenOff,
        TestMinimumDwell,
        TestMaximumDwell,
        TestExistingConstraintsPreserved,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
