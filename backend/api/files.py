"""
backend/api/files.py
---------------------
File management endpoints — allow downloading individual result files.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

RUNS_DIR = Path(__file__).resolve().parent.parent.parent / "runs"

ALLOWED_EXTENSIONS = {".json", ".txt", ".czml", ".geojson"}


@router.get("/{simulation_id}/{path:path}", summary="Download a result file")
async def download_file(simulation_id: str, path: str):
    """Download any output file from a simulation run."""
    file_path = RUNS_DIR / simulation_id / path
    # Security: ensure we stay within the run dir
    try:
        file_path = file_path.resolve()
        RUNS_DIR.resolve()
        file_path.relative_to(RUNS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied.")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=403, detail=f"File type '{file_path.suffix}' not allowed for download.")

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )
