"""
backend/api/simulations.py
---------------------------
Endpoints for simulation lifecycle management.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.models.simulation import SimulationCreate
from backend.services.region_service import get_region, get_region_files
from backend.services.radar_config_service import validate_radar_config, get_config_summary
from backend.workers.simulation_worker import (
    register_simulation, get_simulation, list_simulations, launch_simulation
)

router = APIRouter()

RUNS_DIR = Path(__file__).resolve().parent.parent.parent / "runs"


# ── Create simulation ─────────────────────────────────────────────────────────

@router.post("/", summary="Create and start a new simulation", status_code=201)
async def create_simulation(body: SimulationCreate, background_tasks: BackgroundTasks):
    """
    Validate inputs and launch the pipeline in the background.
    Returns immediately with the simulation ID.
    """
    # 1. Check region exists
    region = get_region(body.region_id)
    if not region:
        raise HTTPException(status_code=404, detail=f"Region '{body.region_id}' not found.")

    # 2. Validate radar config
    valid, errors = validate_radar_config(body.radar_config)
    if not valid:
        raise HTTPException(
            status_code=422,
            detail={"message": "Invalid radar configuration", "errors": errors},
        )

    # 3. Check DEM and GeoJSON availability
    try:
        region_files = get_region_files(body.region_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not region_files.get("geojson"):
        raise HTTPException(
            status_code=422,
            detail=f"GeoJSON file for region '{body.region_id}' is not available."
        )
    if not region_files.get("dem"):
        raise HTTPException(
            status_code=422,
            detail=(
                f"DEM file for region '{body.region_id}' is not available. "
                "Please place the GLO-30 DEM raster in data/regions/{body.region_id}/dem.tif "
                "and update the region manifest. See README for download instructions."
            ),
        )

    # 4. Generate simulation ID
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:8].upper()
    sim_id = f"SIM-{ts}-{uid}"

    # 5. Register and launch
    state = register_simulation(sim_id, body.region_id, region.name, body.radar_config)
    background_tasks.add_task(launch_simulation, state, region_files)

    return {
        "simulation_id": sim_id,
        "status": "started",
        "message": f"Simulation '{sim_id}' launched. Poll /api/simulations/{sim_id}/status for progress.",
    }


# ── List simulations ──────────────────────────────────────────────────────────

@router.get("/", summary="List all simulations")
async def list_all_simulations():
    return {"simulations": list_simulations()}


# ── Simulation status ─────────────────────────────────────────────────────────

@router.get("/{simulation_id}/status", summary="Get simulation status")
async def get_simulation_status(simulation_id: str):
    state = _get_or_404(simulation_id)
    return state.to_dict()


# ── Simulation logs ───────────────────────────────────────────────────────────

@router.get("/{simulation_id}/logs", summary="Get pipeline logs")
async def get_simulation_logs(simulation_id: str, last: int = 500):
    state = _get_or_404(simulation_id)
    return {
        "simulation_id": simulation_id,
        "logs": state.get_full_logs()[-last:] if last else state.get_full_logs(),
    }


# ── Simulation results ────────────────────────────────────────────────────────

@router.get("/{simulation_id}/results", summary="Get simulation results")
async def get_simulation_results(simulation_id: str):
    state = _get_or_404(simulation_id)
    data = state.to_dict()

    if data.get("current_stage") not in ("completed", "failed"):
        return JSONResponse(
            status_code=202,
            content={"message": "Simulation is still running.", "stage": data.get("current_stage")},
        )

    run_dir = RUNS_DIR / simulation_id
    results = {"simulation_id": simulation_id, "region_id": data.get("region_id")}

    # Load placement results
    placement_json = run_dir / "placement" / "radar_meta_optimized.json"
    if placement_json.exists():
        with open(placement_json) as f:
            radars = json.load(f)
        results["num_radars"] = len(radars)
        type_counts = {}
        for r in radars:
            type_counts[r.get("type", "?")] = type_counts.get(r.get("type", "?"), 0) + 1
        results["radar_types"] = type_counts

    # Load coverage results
    matrix_json = run_dir / "coverage" / "c_matrix.json"
    if matrix_json.exists():
        with open(matrix_json) as f:
            c_matrix = json.load(f)
        K = len(c_matrix[0]) if c_matrix else 0
        N = len(c_matrix)
        results["num_boundary_points"] = K
        covered_any = [
            any(c_matrix[i][j] == 1 for i in range(N))
            for j in range(K)
        ]
        covered_count = sum(covered_any)
        results["coverage_pct_any"] = round(100 * covered_count / K, 2) if K else 0
        results["uncoverable_pts"] = K - covered_count

    # Load schedule results
    schedule_json = run_dir / "schedule" / "schedule_output.json"
    if schedule_json.exists():
        with open(schedule_json) as f:
            sched = json.load(f)
        results["phase"] = sched.get("phase")
        results["solver_status"] = sched.get("status")
        results["total_on_slots"] = sched.get("total_on")
        results["total_void"] = sched.get("total_void")
        results["void_pct"] = sched.get("void_pct")
        results["total_overlap"] = sched.get("total_overlap")
        results["overlap_density"] = sched.get("overlap_density")
        results["frequency_bands_used"] = sched.get("frequency_bands_used")

    # Load benchmark results
    baseline_json = run_dir / "benchmark" / "baseline_results.json"
    if baseline_json.exists():
        with open(baseline_json) as f:
            results["baselines"] = json.load(f)

    # Output file URLs
    base_url = f"/runs/{simulation_id}"
    output_files = {}
    file_map = {
        "radar_meta_optimized.json": f"{base_url}/placement/radar_meta_optimized.json",
        "radar_type_config.json":    f"{base_url}/placement/radar_type_config.json",
        "c_matrix.json":             f"{base_url}/coverage/c_matrix.json",
        "radar_meta.json":           f"{base_url}/coverage/radar_meta.json",
        "schedule_output.json":      f"{base_url}/schedule/schedule_output.json",
        "baseline_results.json":     f"{base_url}/benchmark/baseline_results.json",
        "baseline_report.txt":       f"{base_url}/benchmark/baseline_report.txt",
        "radar_coverage.czml":       f"{base_url}/visualization/radar_coverage.czml",
    }
    for name, url in file_map.items():
        local = run_dir / "/".join(url.split("/")[3:])
        if local.exists():
            output_files[name] = url
    results["output_files"] = output_files

    return results


# ── Visualization info ────────────────────────────────────────────────────────

@router.get("/{simulation_id}/visualization", summary="Get visualization asset URLs")
async def get_visualization_info(simulation_id: str):
    state = _get_or_404(simulation_id)
    data = state.to_dict()

    if data.get("current_stage") not in ("completed",):
        raise HTTPException(
            status_code=202,
            detail=f"Simulation is not yet complete (stage: {data.get('current_stage')}).",
        )

    run_dir = RUNS_DIR / simulation_id
    czml_path = run_dir / "visualization" / "radar_coverage.czml"
    border_path = run_dir / "input" / "border.geojson"

    if not czml_path.exists():
        raise HTTPException(status_code=404, detail="CZML file not found.")

    return {
        "simulation_id": simulation_id,
        "czml_url": f"/runs/{simulation_id}/visualization/radar_coverage.czml",
        "border_url": f"/runs/{simulation_id}/input/border.geojson",
        "radar_meta_url": f"/runs/{simulation_id}/visualization/radar_meta.json",
        "schedule_url": f"/runs/{simulation_id}/visualization/schedule_output.json",
        "type_config_url": f"/runs/{simulation_id}/visualization/radar_type_config.json",
    }


# ── WebSocket for live logs ───────────────────────────────────────────────────

@router.websocket("/{simulation_id}/ws")
async def simulation_websocket(websocket: WebSocket, simulation_id: str):
    """Stream live pipeline logs via WebSocket."""
    await websocket.accept()
    import asyncio

    state = get_simulation(simulation_id)
    if not state:
        await websocket.send_json({"error": f"Simulation '{simulation_id}' not found."})
        await websocket.close()
        return

    sent_count = 0
    try:
        while True:
            data = state.to_dict()
            logs = state.get_full_logs()

            # Send new log lines
            new_lines = logs[sent_count:]
            if new_lines:
                await websocket.send_json({"type": "logs", "lines": new_lines})
                sent_count = len(logs)

            # Send status update
            await websocket.send_json({
                "type": "status",
                "current_stage": data.get("current_stage"),
                "stages": data.get("stages", []),
            })

            # Check if done
            if data.get("current_stage") in ("completed", "failed"):
                await websocket.send_json({"type": "done", "stage": data.get("current_stage")})
                break

            await asyncio.sleep(1.5)
    except WebSocketDisconnect:
        pass


# ── Validate radar config ─────────────────────────────────────────────────────

class ValidateConfigRequest(BaseModel):
    config: dict


@router.post("/validate-config", summary="Validate radar configuration")
async def validate_config(body: ValidateConfigRequest):
    valid, errors = validate_radar_config(body.config)
    if not valid:
        return {"valid": False, "errors": errors}
    summary = get_config_summary(body.config)
    return {"valid": True, "errors": [], "summary": summary}


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_or_404(simulation_id: str):
    state = get_simulation(simulation_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Simulation '{simulation_id}' not found.")
    return state
