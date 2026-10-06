"""
backend/workers/simulation_worker.py
--------------------------------------
Background simulation pipeline worker.

Architecture:
  - Each simulation runs in a daemon Thread.
  - Progress is logged to runs/<sim_id>/logs/pipeline.log and also
    appended to an in-memory deque for live WebSocket streaming.
  - Stage transitions are written to runs/<sim_id>/status.json atomically.

Pipeline stages (in order):
  1. validate   — check all inputs are present and valid
  2. placement  — optimize_placement.py (via subprocess wrapper)
  3. coverage   — build_matrix_1.3.py  (via subprocess wrapper)
  4. scheduling — scheduler.py          (via subprocess wrapper)
  5. benchmark  — baseline_comparison.py (via subprocess wrapper)
  6. czml       — generate_czml.py      (via subprocess wrapper)
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

# Project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RUNS_DIR = PROJECT_ROOT / "runs"
SCRIPTS_DIR = PROJECT_ROOT   # original scripts live here
DATA_DIR = PROJECT_ROOT / "data" / "regions"


# ── In-memory simulation registry ─────────────────────────────────────────────
# Maps simulation_id -> SimulationState
_simulations: Dict[str, "SimulationState"] = {}
_lock = threading.Lock()


def register_simulation(sim_id: str, region_id: str, region_name: str,
                         radar_config: dict) -> "SimulationState":
    state = SimulationState(sim_id, region_id, region_name, radar_config)
    with _lock:
        _simulations[sim_id] = state
    return state


def get_simulation(sim_id: str) -> Optional["SimulationState"]:
    # First check memory
    with _lock:
        if sim_id in _simulations:
            return _simulations[sim_id]
    # Fall back to disk (for server restarts)
    return _load_from_disk(sim_id)


def list_simulations() -> list:
    results = []
    # In-memory first
    with _lock:
        known = dict(_simulations)
    for state in known.values():
        results.append(state.to_dict())

    # Also scan runs/ directory for any persisted simulations not in memory
    if RUNS_DIR.exists():
        for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
            if run_dir.is_dir() and run_dir.name not in known:
                loaded = _load_from_disk(run_dir.name)
                if loaded:
                    results.append(loaded.to_dict())
    return results


def _load_from_disk(sim_id: str) -> Optional["SimulationState"]:
    status_path = RUNS_DIR / sim_id / "status.json"
    if not status_path.exists():
        return None
    try:
        with open(status_path) as f:
            data = json.load(f)
        state = SimulationState.__new__(SimulationState)
        state._data = data
        state.sim_id = sim_id
        state._log_buf = deque(maxlen=2000)
        state._lock = threading.Lock()
        return state
    except Exception:
        return None


# ── SimulationState ────────────────────────────────────────────────────────────

STAGE_ORDER = [
    "validating",
    "placement",
    "coverage_matrix",
    "scheduling",
    "benchmarking",
    "czml_generation",
]

STAGE_LABELS = {
    "validating":    "Input Validation",
    "placement":     "Radar Placement Optimization",
    "coverage_matrix": "LOS / Coverage Matrix",
    "scheduling":    "CSP / ILP Scheduling",
    "benchmarking":  "Benchmarking",
    "czml_generation": "CZML Generation",
}


class SimulationState:
    def __init__(self, sim_id: str, region_id: str, region_name: str,
                 radar_config: dict):
        self.sim_id = sim_id
        self._lock = threading.Lock()
        self._log_buf: deque = deque(maxlen=2000)

        self._data = {
            "simulation_id": sim_id,
            "region_id": region_id,
            "region_name": region_name,
            "created_at": _now(),
            "started_at": None,
            "completed_at": None,
            "current_stage": "pending",
            "stages": [
                {"name": s, "label": STAGE_LABELS[s], "status": "waiting",
                 "started_at": None, "completed_at": None, "error": None}
                for s in STAGE_ORDER
            ],
            "error": None,
            "run_dir": str(RUNS_DIR / sim_id),
            "radar_config": radar_config,
        }
        # Persist immediately
        self._persist()

    def to_dict(self) -> dict:
        with self._lock:
            return dict(self._data)

    def start(self):
        with self._lock:
            self._data["started_at"] = _now()
            self._data["current_stage"] = STAGE_ORDER[0]
        self._persist()

    def begin_stage(self, stage: str):
        with self._lock:
            self._data["current_stage"] = stage
            for s in self._data["stages"]:
                if s["name"] == stage:
                    s["status"] = "running"
                    s["started_at"] = _now()
        self._persist()

    def complete_stage(self, stage: str):
        with self._lock:
            for s in self._data["stages"]:
                if s["name"] == stage:
                    s["status"] = "done"
                    s["completed_at"] = _now()
        self._persist()

    def fail_stage(self, stage: str, error: str):
        with self._lock:
            for s in self._data["stages"]:
                if s["name"] == stage:
                    s["status"] = "failed"
                    s["completed_at"] = _now()
                    s["error"] = error
            self._data["current_stage"] = "failed"
            self._data["error"] = error
            self._data["completed_at"] = _now()
        self._persist()

    def mark_completed(self):
        with self._lock:
            self._data["current_stage"] = "completed"
            self._data["completed_at"] = _now()
        self._persist()

    def log(self, line: str):
        with self._lock:
            self._log_buf.append(line)
        # Append to log file
        log_path = RUNS_DIR / self.sim_id / "logs" / "pipeline.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def get_logs(self, last_n: int = 500) -> list:
        with self._lock:
            logs = list(self._log_buf)
        return logs[-last_n:]

    def get_full_logs(self) -> list:
        log_path = RUNS_DIR / self.sim_id / "logs" / "pipeline.log"
        if log_path.exists():
            with open(log_path, encoding="utf-8", errors="replace") as f:
                return f.read().splitlines()
        return []

    def _persist(self):
        run_dir = RUNS_DIR / self.sim_id
        run_dir.mkdir(parents=True, exist_ok=True)
        status_path = run_dir / "status.json"
        with self._lock:
            data = dict(self._data)
        # Don't persist radar_config inline (it's also in input/)
        tmp = dict(data)
        tmp.pop("radar_config", None)
        with open(status_path, "w") as f:
            json.dump(tmp, f, indent=2)


# ── Pipeline worker ────────────────────────────────────────────────────────────

def run_pipeline(state: SimulationState, region_files: dict):
    """
    Entry point for the background thread. Runs all pipeline stages sequentially.
    """
    sim_id = state.sim_id
    run_dir = RUNS_DIR / sim_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Sub-directories
    (run_dir / "input").mkdir(exist_ok=True)
    (run_dir / "placement").mkdir(exist_ok=True)
    (run_dir / "coverage").mkdir(exist_ok=True)
    (run_dir / "schedule").mkdir(exist_ok=True)
    (run_dir / "benchmark").mkdir(exist_ok=True)
    (run_dir / "visualization").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    state.start()
    state.log(f"=== Simulation {sim_id} started at {_now()} ===")

    try:
        # ── Stage 0: Validate ────────────────────────────────────────────────
        _run_stage(state, "validating",
                   lambda: _stage_validate(state, run_dir, region_files))

        # ── Stage 1: Placement ───────────────────────────────────────────────
        _run_stage(state, "placement",
                   lambda: _stage_placement(state, run_dir, region_files))

        # ── Stage 2: Coverage Matrix ─────────────────────────────────────────
        _run_stage(state, "coverage_matrix",
                   lambda: _stage_coverage(state, run_dir, region_files))

        # ── Stage 3: Scheduling ──────────────────────────────────────────────
        _run_stage(state, "scheduling",
                   lambda: _stage_scheduling(state, run_dir))

        # ── Stage 4: Benchmarking ────────────────────────────────────────────
        _run_stage(state, "benchmarking",
                   lambda: _stage_benchmark(state, run_dir))

        # ── Stage 5: CZML Generation ─────────────────────────────────────────
        _run_stage(state, "czml_generation",
                   lambda: _stage_czml(state, run_dir, region_files))

        state.mark_completed()
        state.log(f"=== Simulation {sim_id} completed at {_now()} ===")

    except _StageFailed as e:
        state.log(f"[FATAL] Pipeline aborted: {e}")


def _run_stage(state: SimulationState, stage: str, fn: Callable):
    state.begin_stage(stage)
    state.log(f"\n>>> Starting stage: {STAGE_LABELS[stage]}")
    try:
        fn()
        state.complete_stage(stage)
        state.log(f"<<< Stage complete: {STAGE_LABELS[stage]}")
    except _StageFailed:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        state.log(f"[ERROR] {e}")
        state.log(tb)
        state.fail_stage(stage, str(e))
        raise _StageFailed(str(e)) from e


class _StageFailed(Exception):
    pass


# ── Stage implementations ──────────────────────────────────────────────────────

def _stage_validate(state: SimulationState, run_dir: Path, region_files: dict):
    """Validate inputs and copy them into the run directory."""
    state.log("Validating region files...")

    geojson = region_files.get("geojson")
    dem = region_files.get("dem")

    if not geojson or not Path(geojson).exists():
        raise ValueError(
            f"GeoJSON file not found. Expected at: {geojson}. "
            "Please ensure the region's GeoJSON file is present in data/regions/<id>/."
        )
    state.log(f"  GeoJSON: {geojson} [{Path(geojson).stat().st_size // 1024} KB] ✓")

    if not dem or not Path(dem).exists():
        raise ValueError(
            f"DEM file not found. Expected at: {dem}. "
            "The DEM (Digital Elevation Model) raster file is required for terrain LOS computation. "
            "Please download the GLO-30 DEM for this region and place it in data/regions/<id>/. "
            "See the README for download instructions."
        )
    state.log(f"  DEM:     {dem} [{Path(dem).stat().st_size // 1024 // 1024} MB] ✓")

    # Validate radar config
    radar_config = state._data.get("radar_config", {})
    from backend.services.radar_config_service import validate_radar_config
    valid, errors = validate_radar_config(radar_config)
    if not valid:
        raise ValueError("Radar configuration is invalid:\n" + "\n".join(f"  - {e}" for e in errors))
    state.log(f"  Radar config: {len(radar_config)} type(s), "
              f"{sum(v['count'] for v in radar_config.values())} total radars ✓")

    # Copy input files
    input_dir = run_dir / "input"
    geojson_dest = input_dir / "border.geojson"
    dem_dest = input_dir / "dem.tif"

    shutil.copy2(geojson, geojson_dest)
    state.log(f"  Copied GeoJSON → {geojson_dest}")

    # DEM can be large — use hard link if possible to save space
    try:
        os.link(dem, dem_dest)
        state.log(f"  Linked DEM → {dem_dest}")
    except (OSError, NotImplementedError):
        state.log(f"  Copying DEM (this may take a moment)...")
        shutil.copy2(dem, dem_dest)
        state.log(f"  Copied DEM → {dem_dest}")

    # Write radar config into run dir
    radar_config_path = input_dir / "radar_type_config.json"
    with open(radar_config_path, "w") as f:
        json.dump(radar_config, f, indent=2)
    state.log(f"  Saved radar_type_config.json ✓")


def _stage_placement(state: SimulationState, run_dir: Path, region_files: dict):
    """Run optimize_placement.py via subprocess wrapper."""
    input_dir = run_dir / "input"
    placement_dir = run_dir / "placement"
    ref_point = region_files.get("reference_point")  # [lon, lat]

    # Build the --config argument (radar types JSON)
    radar_config = state._data.get("radar_config", {})

    # Write placement config (the format optimize_placement.py --config expects)
    placement_config_path = placement_dir / "placement_config.json"
    with open(placement_config_path, "w") as f:
        json.dump(radar_config, f, indent=2)

    # Write the wrapper script into the placement dir
    wrapper = _build_placement_wrapper(
        border_file=str(input_dir / "border.geojson"),
        dem_file=str(input_dir / "dem.tif"),
        output_placement=str(placement_dir / "radar_meta_optimized.json"),
        output_type_config=str(placement_dir / "radar_type_config.json"),
        config_file=str(placement_config_path),
        reference_point=ref_point,
        original_script=str(SCRIPTS_DIR / "optimize_placement.py"),
    )
    wrapper_path = placement_dir / "_run_placement.py"
    wrapper_path.write_text(wrapper, encoding="utf-8")

    _run_subprocess(state, [sys.executable, "-u", str(wrapper_path)], cwd=str(placement_dir))

    # Verify outputs
    _require_file(placement_dir / "radar_meta_optimized.json", "radar_meta_optimized.json")
    _require_file(placement_dir / "radar_type_config.json", "radar_type_config.json")
    state.log("  Placement outputs verified ✓")


def _stage_coverage(state: SimulationState, run_dir: Path, region_files: dict):
    """Run build_matrix_1.3.py via subprocess wrapper."""
    input_dir = run_dir / "input"
    placement_dir = run_dir / "placement"
    coverage_dir = run_dir / "coverage"

    # build_matrix needs these files in its CWD OR at explicit paths
    # We write a wrapper that patches the module-level constants
    wrapper = _build_matrix_wrapper(
        border_file=str(input_dir / "border.geojson"),
        dem_file=str(input_dir / "dem.tif"),
        input_placement=str(placement_dir / "radar_meta_optimized.json"),
        input_type_config=str(placement_dir / "radar_type_config.json"),
        output_matrix=str(coverage_dir / "c_matrix.json"),
        output_meta=str(coverage_dir / "radar_meta.json"),
        original_script=str(SCRIPTS_DIR / "build_matrix_1.3.py"),
    )
    wrapper_path = coverage_dir / "_run_build_matrix.py"
    wrapper_path.write_text(wrapper, encoding="utf-8")

    _run_subprocess(state, [sys.executable, "-u", str(wrapper_path)], cwd=str(coverage_dir))

    _require_file(coverage_dir / "c_matrix.json", "c_matrix.json")
    _require_file(coverage_dir / "radar_meta.json", "radar_meta.json")
    state.log("  Coverage matrix outputs verified ✓")


def _stage_scheduling(state: SimulationState, run_dir: Path):
    """Run scheduler.py via subprocess wrapper."""
    placement_dir = run_dir / "placement"
    coverage_dir = run_dir / "coverage"
    schedule_dir = run_dir / "schedule"

    wrapper = _build_scheduler_wrapper(
        matrix_file=str(coverage_dir / "c_matrix.json"),
        meta_file=str(coverage_dir / "radar_meta.json"),
        type_config_file=str(placement_dir / "radar_type_config.json"),
        output_file=str(schedule_dir / "schedule_output.json"),
        original_script=str(SCRIPTS_DIR / "scheduler.py"),
    )
    wrapper_path = schedule_dir / "_run_scheduler.py"
    wrapper_path.write_text(wrapper, encoding="utf-8")

    _run_subprocess(state, [sys.executable, "-u", str(wrapper_path)], cwd=str(schedule_dir),
                    timeout=700)  # CP-SAT can take a while

    _require_file(schedule_dir / "schedule_output.json", "schedule_output.json")
    state.log("  Schedule output verified ✓")


def _stage_benchmark(state: SimulationState, run_dir: Path):
    """Run baseline_comparison.py via subprocess wrapper."""
    placement_dir = run_dir / "placement"
    coverage_dir = run_dir / "coverage"
    schedule_dir = run_dir / "schedule"
    benchmark_dir = run_dir / "benchmark"

    wrapper = _build_baseline_wrapper(
        matrix_file=str(coverage_dir / "c_matrix.json"),
        meta_file=str(coverage_dir / "radar_meta.json"),
        type_config_file=str(placement_dir / "radar_type_config.json"),
        csp_output_file=str(schedule_dir / "schedule_output.json"),
        out_json=str(benchmark_dir / "baseline_results.json"),
        out_txt=str(benchmark_dir / "baseline_report.txt"),
        original_script=str(SCRIPTS_DIR / "baseline_comparison.py"),
    )
    wrapper_path = benchmark_dir / "_run_baseline.py"
    wrapper_path.write_text(wrapper, encoding="utf-8")

    _run_subprocess(state, [sys.executable, "-u", str(wrapper_path)], cwd=str(benchmark_dir))

    _require_file(benchmark_dir / "baseline_results.json", "baseline_results.json")
    state.log("  Benchmark outputs verified ✓")


def _stage_czml(state: SimulationState, run_dir: Path, region_files: dict):
    """Run generate_czml.py (already has CLI args — call directly)."""
    input_dir = run_dir / "input"
    placement_dir = run_dir / "placement"
    coverage_dir = run_dir / "coverage"
    schedule_dir = run_dir / "schedule"
    viz_dir = run_dir / "visualization"

    czml_out = viz_dir / "radar_coverage.czml"

    cmd = [
        sys.executable, "-u",
        str(SCRIPTS_DIR / "generate_czml.py"),
        "--radar-meta",    str(coverage_dir / "radar_meta.json"),
        "--c-matrix",      str(coverage_dir / "c_matrix.json"),
        "--schedule",      str(schedule_dir / "schedule_output.json"),
        "--border",        str(input_dir / "border.geojson"),
        "--type-config",   str(placement_dir / "radar_type_config.json"),
        "--out",           str(czml_out),
    ]
    _run_subprocess(state, cmd, cwd=str(viz_dir))

    _require_file(czml_out, "radar_coverage.czml")

    # Also copy key result files into visualization dir for easy access
    for src, dst_name in [
        (coverage_dir / "radar_meta.json", "radar_meta.json"),
        (coverage_dir / "c_matrix.json", "c_matrix.json"),
        (schedule_dir / "schedule_output.json", "schedule_output.json"),
        (placement_dir / "radar_type_config.json", "radar_type_config.json"),
        (run_dir / "benchmark" / "baseline_results.json", "baseline_results.json"),
    ]:
        if src.exists():
            shutil.copy2(src, viz_dir / dst_name)

    state.log(f"  CZML generated: {czml_out} [{czml_out.stat().st_size // 1024} KB] ✓")


# ── Subprocess runner ─────────────────────────────────────────────────────────

def _run_subprocess(state: SimulationState, cmd: list, cwd: str, timeout: int = 600):
    """Run a subprocess, streaming stdout/stderr to state.log()."""
    state.log(f"  $ {' '.join(str(c) for c in cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        for line in proc.stdout:
            state.log(line.rstrip())
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError(f"Stage timed out after {timeout}s.")

    if proc.returncode != 0:
        raise RuntimeError(f"Stage subprocess exited with code {proc.returncode}. "
                           "Check the pipeline logs for details.")


def _require_file(path: Path, name: str):
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Expected output file '{name}' was not generated at {path}. "
            "The stage may have failed silently — check the pipeline logs."
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Wrapper script builders ────────────────────────────────────────────────────

def _build_placement_wrapper(border_file, dem_file, output_placement, output_type_config,
                              config_file, reference_point, original_script) -> str:
    """
    Build a Python wrapper that patches optimize_placement.py's module-level
    constants before running main().
    """
    ref_arg = ""
    if reference_point:
        lon, lat = reference_point
        ref_arg = f", \"--reference-point\", \"{lon},{lat}\""

    return f"""
import sys
import os

# Override constants BEFORE the module is executed
# We do this by patching sys.argv to use --config, and by using importlib
sys.argv = [
    "optimize_placement.py",
    "--config", {config_file!r}{ref_arg}
]

# We can't import optimize_placement.py because its main() is triggered
# by if __name__ == "__main__". Instead, run it in a patched namespace.
import importlib.util, types

spec = importlib.util.spec_from_file_location(
    "optimize_placement",
    {original_script!r}
)

# Patch constants by executing the module with overridden globals
with open({original_script!r}, encoding="utf-8") as _f:
    _src = _f.read()

# Inject overrides at the top
_overrides = \"\"\"
BORDER_FILE = {border_file!r}
DEM_FILE    = {dem_file!r}
OUTPUT_PLACEMENT   = {output_placement!r}
OUTPUT_TYPE_CONFIG = {output_type_config!r}
\"\"\"

# Execute: we replace the const block by prepending an override block
# that re-assigns after the original definitions
_full_src = _src + "\\n" + _overrides + "\\nmain()\\n"
exec(compile(_full_src, {original_script!r}, "exec"), {{"__name__": "__main__", "__file__": {original_script!r}}})
"""


def _build_matrix_wrapper(border_file, dem_file, input_placement, input_type_config,
                           output_matrix, output_meta, original_script) -> str:
    """
    Build a wrapper for build_matrix_1.3.py.
    Since that script runs Steps 1-6 at module level, we exec() it with
    patched constants prepended.
    """
    return f"""
import sys, os

with open({original_script!r}, encoding="utf-8") as _f:
    _src = _f.read()

_overrides = \"\"\"
BORDER_FILE   = {border_file!r}
DEM_FILE      = {dem_file!r}
INPUT_PLACEMENT   = {input_placement!r}
INPUT_TYPE_CONFIG = {input_type_config!r}
OUTPUT_MATRIX = {output_matrix!r}
OUTPUT_META   = {output_meta!r}
\"\"\"

# Append overrides AFTER the const block; Python executes top-to-bottom
# so later assignments win. The module-level steps (1-6) then use the
# overridden values.
_full_src = _src + "\\n" + _overrides
exec(compile(_full_src, {original_script!r}, "exec"), {{"__name__": "__main__", "__file__": {original_script!r}}})
"""


def _build_scheduler_wrapper(matrix_file, meta_file, type_config_file,
                              output_file, original_script) -> str:
    return f"""
import sys, os

with open({original_script!r}, encoding="utf-8") as _f:
    _src = _f.read()

_overrides = \"\"\"
MATRIX_FILE      = {matrix_file!r}
META_FILE        = {meta_file!r}
TYPE_CONFIG_FILE = {type_config_file!r}

import json as _json
_output = None
\"\"\"

# The scheduler writes to "schedule_output.json" (relative CWD) in its main block.
# We additionally add a post-exec step to copy to our target path.
_post = \"\"\"
import shutil as _shutil, os as _os
_expected = _os.path.join(_os.path.dirname({original_script!r}), "schedule_output.json")
_cwd_expected = "schedule_output.json"
if _os.path.exists(_cwd_expected):
    _shutil.copy2(_cwd_expected, {output_file!r})
\"\"\"

_full_src = _src + "\\n" + _overrides + "\\n" + _post
exec(compile(_full_src, {original_script!r}, "exec"), {{"__name__": "__main__", "__file__": {original_script!r}}})
"""


def _build_baseline_wrapper(matrix_file, meta_file, type_config_file,
                             csp_output_file, out_json, out_txt,
                             original_script) -> str:
    return f"""
import sys, os, shutil

with open({original_script!r}, encoding="utf-8") as _f:
    _src = _f.read()

_overrides = \"\"\"
MATRIX_FILE      = {matrix_file!r}
META_FILE        = {meta_file!r}
TYPE_CONFIG_FILE = {type_config_file!r}
CSP_OUTPUT_FILE  = {csp_output_file!r}
\"\"\"

_post = \"\"\"
import shutil as _sh, os as _os
if _os.path.exists("baseline_results.json"):
    _sh.copy2("baseline_results.json", {out_json!r})
if _os.path.exists("baseline_report.txt"):
    _sh.copy2("baseline_report.txt", {out_txt!r})
\"\"\"

_full_src = _src + "\\n" + _overrides + "\\n" + _post
exec(compile(_full_src, {original_script!r}, "exec"), {{"__name__": "__main__", "__file__": {original_script!r}}})
"""


# ── Launch helper ──────────────────────────────────────────────────────────────

def launch_simulation(state: SimulationState, region_files: dict):
    """Start the pipeline in a background daemon thread."""
    t = threading.Thread(
        target=run_pipeline,
        args=(state, region_files),
        daemon=True,
        name=f"sim-{state.sim_id}",
    )
    t.start()
    return t
