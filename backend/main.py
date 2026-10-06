"""
backend/main.py
---------------
FastAPI application entry point for the Border Radar CSP Simulation Platform.
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure the project root (where original scripts live) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.api import regions, simulations, files

app = FastAPI(
    title="Border Radar CSP Simulation Platform",
    description="End-to-end geospatial radar optimization and scheduling",
    version="1.0.0",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routers ───────────────────────────────────────────────────────────────
app.include_router(regions.router, prefix="/api/regions", tags=["regions"])
app.include_router(simulations.router, prefix="/api/simulations", tags=["simulations"])
app.include_router(files.router, prefix="/api/files", tags=["files"])

# ── Serve generated output files (CZML, JSON results) ─────────────────────
RUNS_DIR = PROJECT_ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)
app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")

# ── Serve region data files (GeoJSON etc.) ───────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


@app.get("/viewer.html", tags=["viewer"])
async def serve_viewer():
    viewer_path = PROJECT_ROOT / "viewer.html"
    if not viewer_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="viewer.html not found")
    return FileResponse(str(viewer_path), media_type="text/html")


@app.get("/", tags=["health"])
async def root():
    return {"status": "ok", "service": "Border Radar CSP Simulation Platform"}


@app.get("/health", tags=["health"])
async def health():
    return {"status": "healthy"}
