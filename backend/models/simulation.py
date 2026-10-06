"""
backend/models/simulation.py
-----------------------------
Pydantic models for simulation lifecycle.
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class SimulationStage(str, Enum):
    PENDING            = "pending"
    VALIDATING         = "validating"
    PLACEMENT          = "placement"
    COVERAGE_MATRIX    = "coverage_matrix"
    SCHEDULING         = "scheduling"
    BENCHMARKING       = "benchmarking"
    CZML_GENERATION    = "czml_generation"
    COMPLETED          = "completed"
    FAILED             = "failed"


class StageStatus(str, Enum):
    WAITING    = "waiting"
    RUNNING    = "running"
    DONE       = "done"
    FAILED     = "failed"
    SKIPPED    = "skipped"


class PipelineStageInfo(BaseModel):
    name: str
    label: str
    status: StageStatus = StageStatus.WAITING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class SimulationCreate(BaseModel):
    region_id: str
    radar_config: Dict[str, Any]   # radar_type_config.json content


class SimulationStatus(BaseModel):
    simulation_id: str
    region_id: str
    region_name: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    current_stage: SimulationStage
    stages: List[PipelineStageInfo]
    error: Optional[str] = None
    run_dir: Optional[str] = None


class SimulationResults(BaseModel):
    simulation_id: str
    region_id: str
    # Placement
    num_radars: Optional[int] = None
    radar_types: Optional[Dict[str, int]] = None
    # Coverage
    num_boundary_points: Optional[int] = None
    coverage_pct_any: Optional[float] = None
    uncoverable_pts: Optional[int] = None
    # Scheduling
    phase: Optional[int] = None
    solver_status: Optional[str] = None
    total_on_slots: Optional[int] = None
    total_void: Optional[int] = None
    void_pct: Optional[float] = None
    total_overlap: Optional[int] = None
    overlap_density: Optional[float] = None
    frequency_bands_used: Optional[List[str]] = None
    # Benchmarks
    baselines: Optional[Dict[str, Any]] = None
    # Files
    output_files: Optional[Dict[str, str]] = None  # name -> URL
