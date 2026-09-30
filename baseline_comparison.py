"""
baseline_comparison.py  —  Baseline vs CSP Scheduler Benchmarking
==================================================================

Compares the Two-Phase CSP/ILP Radar Scheduler (scheduler.py) against
three reference baselines, all evaluated on the same:
  - c_matrix.json          (coverage matrix)
  - radar_meta.json        (radar metadata)
  - radar_type_config.json (per-type energy budget & cooling constraints)

Baselines
---------
  B1 – Always-ON (Naive)
       Every radar is ON for every time slot.  Violates energy/cooling budgets
       intentionally — represents the theoretical upper-bound of coverage.

  B2 – Random Duty-Cycling (stochastic)
       Each radar is switched ON independently with probability p = B_i / M,
       i.e. it respects the energy budget in expectation.  Averaged over 50
       independent random trials (seeds 0–49).

  B3 – Greedy Coverage Scheduler (deterministic)
       For each time slot t, activate the subset of radars that greedily
       maximises the number of newly uncovered points in that slot, subject
       to each radar's remaining energy budget and cooling constraints.

CSP result is loaded from schedule_output.json (pre-computed by scheduler.py).

Metrics reported (matching the CSP scheduler output format)
-----------------------------------------------------------
  • total_on          – Σ_{i,t} X[i,t]   (radar-slot activations)
  • total_void        – Σ_{j,t} V[j,t]   (space-time cells uncovered)
  • void_pct          – total_void / (K*M) × 100
  • covered_pts_any   – boundary points covered in ≥1 slot
  • coverage_pct_any  – covered_pts_any / K × 100
  • avg_slot_coverage – mean fraction of K covered per slot
  • total_overlap     – Σ_{j,t} max(0, cover(j,t)-1)  (redundant coverages)
  • overlap_density   – total_overlap / (K*M)
  • energy_violations – radars exceeding their budget   (budget compliance check)
  • cooling_violations– rolling-window violations       (constraint compliance)

Usage
-----
    python baseline_comparison.py

Outputs
-------
  baseline_results.json   — full numeric results for all methods
  baseline_report.txt     — human-readable comparison table (paper-ready)
"""

import json
import random
import sys
import time
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── File paths ────────────────────────────────────────────────────────────────
MATRIX_FILE      = "c_matrix.json"
META_FILE        = "radar_meta.json"
TYPE_CONFIG_FILE = "radar_type_config.json"
CSP_OUTPUT_FILE  = "schedule_output.json"

NUM_TIME_SLOTS   = 24   # M
RANDOM_TRIALS    = 50   # for B2 averaging
RANDOM_SEED_BASE = 0

# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(title):
    print(f"\n{'═'*62}")
    print(f"  {title}")
    print(f"{'═'*62}")


def compute_metrics(sched, C_mat, N, K, M, budgets, L_list, C_list):
    """
    Compute all comparison metrics from a binary schedule matrix sched[N][M].
    Returns a dict of metrics.
    """
    # ── Void / coverage ──────────────────────────────────────────────────────
    total_void = 0
    total_on   = 0
    total_overlap = 0
    slot_coverage_fracs = []
    covered_any = [False] * K   # covered in at least one slot

    for t in range(M):
        covered_in_slot = 0
        for j in range(K):
            cov = sum(C_mat[i][j] * sched[i][t] for i in range(N))
            if cov == 0:
                total_void += 1
            else:
                covered_in_slot += 1
                covered_any[j] = True
                if cov > 1:
                    total_overlap += (cov - 1)
        slot_coverage_fracs.append(covered_in_slot / K)

    for i in range(N):
        total_on += sum(sched[i])

    void_pct         = 100 * total_void / (K * M) if K * M > 0 else 0
    covered_pts_any  = sum(covered_any)
    coverage_pct_any = 100 * covered_pts_any / K if K > 0 else 0
    avg_slot_cov     = 100 * sum(slot_coverage_fracs) / M if M > 0 else 0
    overlap_density  = total_overlap / (K * M) if K * M > 0 else 0

    # ── Energy violations ─────────────────────────────────────────────────────
    energy_violations = sum(
        1 for i in range(N) if sum(sched[i]) > budgets[i]
    )

    # ── Cooling violations (per-radar rolling window) ─────────────────────────
    cooling_violations = 0
    for i in range(N):
        window = L_list[i] + C_list[i]
        if window > 1:
            for t in range(M):
                end = min(t + window, M)
                if sum(sched[i][k] for k in range(t, end)) > L_list[i]:
                    cooling_violations += 1
                    break  # count each radar at most once

    return {
        "total_on":           total_on,
        "total_void":         total_void,
        "void_pct":           void_pct,
        "covered_pts_any":    covered_pts_any,
        "coverage_pct_any":   coverage_pct_any,
        "avg_slot_coverage":  avg_slot_cov,
        "total_overlap":      total_overlap,
        "overlap_density":    overlap_density,
        "energy_violations":  energy_violations,
        "cooling_violations": cooling_violations,
    }


# ── Baseline 1 — Always-ON ────────────────────────────────────────────────────

def baseline_always_on(N, M):
    """Every radar ON at every slot."""
    return [[1] * M for _ in range(N)]


# ── Baseline 2 — Random Duty-Cycling ─────────────────────────────────────────

def baseline_random_single(N, M, budgets, seed):
    """
    One random trial: radar i is ON at each slot independently with
    probability p_i = budgets[i] / M.
    """
    rng = random.Random(seed)
    sched = []
    for i in range(N):
        p = budgets[i] / M
        row = [1 if rng.random() < p else 0 for _ in range(M)]
        sched.append(row)
    return sched


def baseline_random_averaged(N, K, M, C_mat, budgets, L_list, C_list, trials, seed_base):
    """
    Run `trials` random schedules and average metrics.
    Also return the best single trial (by total_void).
    """
    all_metrics = []
    best_void = float("inf")
    best_sched = None

    for t in range(trials):
        sched   = baseline_random_single(N, M, budgets, seed_base + t)
        metrics = compute_metrics(sched, C_mat, N, K, M, budgets, L_list, C_list)
        all_metrics.append(metrics)
        if metrics["total_void"] < best_void:
            best_void  = metrics["total_void"]
            best_sched = sched

    # Average
    avg = {}
    keys = all_metrics[0].keys()
    for k in keys:
        vals = [m[k] for m in all_metrics]
        avg[f"{k}_mean"] = sum(vals) / len(vals)
        avg[f"{k}_std"]  = (sum((v - avg[f"{k}_mean"])**2 for v in vals) / len(vals)) ** 0.5
        avg[f"{k}_min"]  = min(vals)
        avg[f"{k}_max"]  = max(vals)

    return avg, all_metrics, best_sched


# ── Baseline 3 — Greedy Coverage Scheduler ────────────────────────────────────

def baseline_greedy(N, K, M, C_mat, budgets, L_list, C_list):
    """
    Greedy per-slot scheduler with energy budget and cooling enforcement.

    Algorithm:
      For each time slot t (in order):
        1. Build the candidate set of radars that:
           a) still have remaining energy (budget not exhausted), AND
           b) satisfy their cooling constraint (adding an ON-slot here
              would not violate any rolling window starting at t-window+1 .. t).
        2. Greedily add radars one-by-one (in descending order of marginal
           coverage gain) until no further gain is possible.
    """
    sched        = [[0] * M for _ in range(N)]
    remaining_B  = list(budgets)           # remaining energy per radar

    for t in range(M):
        # Determine which radars are cooling-eligible at slot t
        eligible = []
        for i in range(N):
            if remaining_B[i] <= 0:
                continue
            window = L_list[i] + C_list[i]
            if window <= 1:
                eligible.append(i)
                continue
            # Check all windows that include slot t
            ok = True
            for w_start in range(max(0, t - window + 1), t + 1):
                w_end = min(w_start + window, M)
                on_in_window = sum(sched[i][k] for k in range(w_start, w_end))
                # If we were to turn ON at t, on_in_window += 1
                if on_in_window + 1 > L_list[i]:
                    ok = False
                    break
            if ok:
                eligible.append(i)

        # Greedy selection: add radar with highest marginal coverage
        # until no further gain
        activated = set()
        covered   = [False] * K      # covered in this slot so far

        # Pre-initialise covered from already-selected radars (none yet)
        for j in range(K):
            if any(C_mat[ia][j] for ia in activated):
                covered[j] = True

        changed = True
        while changed and eligible:
            changed = False
            best_gain   = 0
            best_radar  = -1
            remaining_eligible = [i for i in eligible if i not in activated]

            for i in remaining_eligible:
                gain = sum(
                    1 for j in range(K)
                    if C_mat[i][j] and not covered[j]
                )
                if gain > best_gain:
                    best_gain  = gain
                    best_radar = i

            if best_radar >= 0 and best_gain > 0:
                activated.add(best_radar)
                for j in range(K):
                    if C_mat[best_radar][j]:
                        covered[j] = True
                changed = True

        # Apply activations
        for i in activated:
            sched[i][t]    = 1
            remaining_B[i] -= 1

    return sched


# ── Load CSP result ───────────────────────────────────────────────────────────

def load_csp_result(path):
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found — CSP results will be marked N/A.")
        return None
    with open(path) as f:
        return json.load(f)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    banner("BASELINE COMPARISON — Radar Scheduler Benchmarking")

    # ── Load inputs ──────────────────────────────────────────────────────────
    print(f"\n[1] Loading {MATRIX_FILE} ...")
    with open(MATRIX_FILE) as f:
        C_mat = json.load(f)
    N = len(C_mat)
    K = len(C_mat[0])
    M = NUM_TIME_SLOTS
    print(f"    N={N} radars,  K={K} boundary points,  M={M} time slots")

    print(f"[2] Loading {META_FILE} ...")
    with open(META_FILE) as f:
        meta = json.load(f)

    print(f"[3] Loading {TYPE_CONFIG_FILE} ...")
    with open(TYPE_CONFIG_FILE) as f:
        type_config = json.load(f)

    # Derive budgets + cooling (same logic as scheduler.py)
    budgets, L_list, C_list = [], [], []
    for r in meta:
        cfg = type_config[r["type"]]
        budgets.append(min(int(round(cfg["energy_budget"])), M))
        L_list.append(int(cfg["cooling_L"]))
        C_list.append(int(cfg["cooling_C"]))

    print(f"    Budgets : {budgets}")
    print(f"    L_list  : {L_list}")
    print(f"    C_list  : {C_list}")

    # ── Load CSP result ──────────────────────────────────────────────────────
    print(f"\n[4] Loading CSP result from {CSP_OUTPUT_FILE} ...")
    csp_raw = load_csp_result(CSP_OUTPUT_FILE)

    results = {}

    # ── CSP Metrics (from saved output) ─────────────────────────────────────
    if csp_raw and "schedule" in csp_raw:
        print("    Computing metrics for CSP scheduler ...")
        csp_sched   = csp_raw["schedule"]
        csp_metrics = compute_metrics(csp_sched, C_mat, N, K, M, budgets, L_list, C_list)
        csp_metrics["phase"]  = csp_raw.get("phase", "?")
        csp_metrics["status"] = csp_raw.get("status", "?")
        results["CSP_scheduler"] = csp_metrics
        print(f"    CSP: phase={csp_metrics['phase']}, status={csp_metrics['status']}, "
              f"void_pct={csp_metrics['void_pct']:.2f}%")
    else:
        results["CSP_scheduler"] = None
        print("    CSP result unavailable.")

    # ── Baseline 1: Always-ON ────────────────────────────────────────────────
    print("\n[5] Running Baseline 1: Always-ON ...")
    t0 = time.time()
    b1_sched   = baseline_always_on(N, M)
    b1_metrics = compute_metrics(b1_sched, C_mat, N, K, M, budgets, L_list, C_list)
    b1_elapsed = time.time() - t0
    b1_metrics["runtime_s"] = b1_elapsed
    results["B1_always_on"] = b1_metrics
    print(f"    Done in {b1_elapsed:.3f}s — void_pct={b1_metrics['void_pct']:.2f}%  "
          f"energy_violations={b1_metrics['energy_violations']}")

    # ── Baseline 2: Random Duty-Cycling ─────────────────────────────────────
    print(f"\n[6] Running Baseline 2: Random Duty-Cycling ({RANDOM_TRIALS} trials) ...")
    t0 = time.time()
    b2_avg, b2_all_metrics, b2_best_sched = baseline_random_averaged(
        N, K, M, C_mat, budgets, L_list, C_list, RANDOM_TRIALS, RANDOM_SEED_BASE
    )
    b2_elapsed = time.time() - t0
    b2_avg["runtime_s"] = b2_elapsed
    results["B2_random"] = b2_avg
    print(f"    Done in {b2_elapsed:.3f}s — "
          f"void_pct_mean={b2_avg['void_pct_mean']:.2f}% "
          f"±{b2_avg['void_pct_std']:.2f}%")

    # ── Baseline 3: Greedy Coverage Scheduler ───────────────────────────────
    print("\n[7] Running Baseline 3: Greedy Coverage Scheduler ...")
    t0 = time.time()
    b3_sched   = baseline_greedy(N, K, M, C_mat, budgets, L_list, C_list)
    b3_elapsed = time.time() - t0
    b3_metrics = compute_metrics(b3_sched, C_mat, N, K, M, budgets, L_list, C_list)
    b3_metrics["runtime_s"] = b3_elapsed
    results["B3_greedy"] = b3_metrics
    print(f"    Done in {b3_elapsed:.3f}s — void_pct={b3_metrics['void_pct']:.2f}%  "
          f"energy_violations={b3_metrics['energy_violations']}")

    # ── Save JSON results ────────────────────────────────────────────────────
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved: baseline_results.json")

    # ── Print comparison table ───────────────────────────────────────────────
    banner("COMPARISON TABLE")

    def fmt(v, pct=False, decimals=2):
        if v is None:
            return "  N/A  "
        if pct:
            return f"{v:8.2f}%"
        return f"{v:{10}.{decimals}f}"

    def get(d, key, mean=False):
        if d is None:
            return None
        k = key + "_mean" if mean else key
        return d.get(k)

    csp  = results.get("CSP_scheduler")
    b1   = results["B1_always_on"]
    b2   = results["B2_random"]
    b3   = results["B3_greedy"]

    lines = []
    lines.append("")
    lines.append(f"  {'Metric':<28} {'B1 Always-ON':>14} {'B2 Random (mean±std)':>22} {'B3 Greedy':>14} {'CSP (proposed)':>16}")
    lines.append(f"  {'─'*28} {'─'*14} {'─'*22} {'─'*14} {'─'*16}")

    def row(label, key, pct=False, dec=2):
        b1v  = get(b1, key)
        b2v  = get(b2, key, mean=True)
        b2sd = get(b2, key + "_std")
        b3v  = get(b3, key)
        cv   = get(csp, key)
        suf  = "%" if pct else ""
        def fv(v, d=dec):
            return f"{v:.{d}f}" if v is not None else "N/A"
        b2str = f"{fv(b2v)}±{fv(b2sd)}" if b2v is not None else "N/A"
        return f"  {label:<28} {fv(b1v)+suf:>14} {b2str+suf:>22} {fv(b3v)+suf:>14} {fv(cv)+suf:>16}"

    lines.append(row("Total ON-slots",         "total_on",           dec=0))
    lines.append(row("Total void (pt×slot)",   "total_void",         dec=0))
    lines.append(row("Void density (%)",        "void_pct",           pct=True))
    lines.append(row("Pts covered (any slot)",  "covered_pts_any",    dec=0))
    lines.append(row("Coverage % (any slot)",   "coverage_pct_any",   pct=True))
    lines.append(row("Avg slot coverage (%)",   "avg_slot_coverage",  pct=True))
    lines.append(row("Total overlap",           "total_overlap",      dec=0))
    lines.append(row("Overlap density",         "overlap_density",    dec=4))
    lines.append(row("Energy violations",       "energy_violations",  dec=0))
    lines.append(row("Cooling violations",      "cooling_violations", dec=0))

    # CSP-specific info
    if csp:
        lines.append(f"\n  CSP phase run  : {csp.get('phase', '?')} "
                     f"(1=min-void, 2=min-overlap)")
        lines.append(f"  CSP status     : {csp.get('status', '?')}")

    lines.append(f"\n  B2 trials      : {RANDOM_TRIALS}  (seeds {RANDOM_SEED_BASE}–{RANDOM_SEED_BASE+RANDOM_TRIALS-1})")
    lines.append(f"  B3 runtime     : {b3_metrics['runtime_s']:.3f}s")
    lines.append(f"  B1 runtime     : {b1_metrics['runtime_s']:.4f}s")

    # ── Improvement over best baseline ───────────────────────────────────────
    if csp:
        lines.append("\n  Improvement of CSP over baselines (void_pct)")
        b2_void_mean = get(b2, "void_pct", mean=True)
        for bname, bvoid in [("B1 Always-ON", get(b1, "void_pct")),
                              ("B2 Random",    b2_void_mean),
                              ("B3 Greedy",    get(b3, "void_pct"))]:
            if bvoid is not None:
                delta = bvoid - get(csp, "void_pct")
                pct_imp = 100 * delta / bvoid if bvoid > 0 else 0
                lines.append(f"    vs {bname:<14}: CSP void_pct is {delta:+.2f}pp  "
                              f"({pct_imp:+.1f}% relative reduction)")

    table_str = "\n".join(lines)
    print(table_str)

    # ── Save human-readable report ────────────────────────────────────────────
    with open("baseline_report.txt", "w", encoding="utf-8") as f:
        f.write("RADAR SCHEDULER BASELINE COMPARISON REPORT\n")
        f.write("=" * 65 + "\n")
        f.write(f"Problem: N={N} radars, K={K} boundary points, M={M} time slots\n")
        f.write(f"Input files: {MATRIX_FILE}, {META_FILE}, {TYPE_CONFIG_FILE}\n\n")
        f.write(table_str)
        f.write("\n\nRAW PER-RADAR SCHEDULE DETAILS\n")
        f.write("─" * 65 + "\n")
        schedules = {
            "B1_always_on": b1_sched,
            "B2_random_best": b2_best_sched,
            "B3_greedy": b3_sched,
        }
        if csp and "schedule" in csp_raw:
            schedules["CSP_scheduler"] = csp_raw["schedule"]

        for name, sched in schedules.items():
            f.write(f"\n{name}:\n")
            header = "                    " + " ".join(f"{t:2d}" for t in range(M))
            f.write(header + "\n")
            for i in range(N):
                on   = sum(sched[i])
                rname = meta[i]["name"] if meta else f"R{i+1:02d}"
                row_s = "  ".join(str(sched[i][t]) for t in range(M))
                f.write(f"  {rname:14s} [{on:2d}/{budgets[i]}]  {row_s}\n")

    print("\n  Saved: baseline_report.txt")
    banner("DONE")
