"""
backend/services/region_service.py
------------------------------------
Discovers and serves region data from data/regions/.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

from backend.models.region import RegionMetadata, RegionStatus

# Root data directory: data/regions/<region_id>/
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "regions"


def get_region_dir(region_id: str) -> Path:
    return DATA_DIR / region_id


def list_regions() -> List[RegionStatus]:
    """Scan data/regions/ and return status for all discovered regions."""
    if not DATA_DIR.exists():
        return []

    results = []
    for region_dir in sorted(DATA_DIR.iterdir()):
        if not region_dir.is_dir():
            continue
        manifest_path = region_dir / "region.json"
        if not manifest_path.exists():
            continue
        try:
            meta = _load_manifest(manifest_path)
            status = _build_status(meta, region_dir)
            results.append(status)
        except Exception as e:
            # Skip malformed manifests rather than crashing the whole listing
            print(f"[region_service] Skipping {region_dir.name}: {e}")
    return results


def get_region(region_id: str) -> Optional[RegionStatus]:
    """Return status for a single region, or None if not found."""
    region_dir = get_region_dir(region_id)
    manifest_path = region_dir / "region.json"
    if not manifest_path.exists():
        return None
    try:
        meta = _load_manifest(manifest_path)
        return _build_status(meta, region_dir)
    except Exception:
        return None


def get_region_files(region_id: str) -> dict:
    """
    Return absolute paths to region files.
    Returns: {geojson: str|None, dem: str|None}
    """
    region_dir = get_region_dir(region_id)
    manifest_path = region_dir / "region.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Region '{region_id}' not found.")

    with open(manifest_path) as f:
        meta_dict = json.load(f)

    geojson = None
    if meta_dict.get("geojson"):
        p = region_dir / meta_dict["geojson"]
        geojson = str(p) if p.exists() else None

    dem = None
    if meta_dict.get("dem"):
        p = region_dir / meta_dict["dem"]
        dem = str(p) if p.exists() else None

    return {
        "geojson": geojson,
        "dem": dem,
        "reference_point": meta_dict.get("reference_point"),
        "region_dir": str(region_dir),
    }


def _load_manifest(manifest_path: Path) -> RegionMetadata:
    with open(manifest_path) as f:
        data = json.load(f)
    return RegionMetadata(**data)


def _build_status(meta: RegionMetadata, region_dir: Path) -> RegionStatus:
    geojson_path = region_dir / meta.geojson if meta.geojson else None
    dem_path = region_dir / meta.dem if meta.dem else None

    geojson_available = geojson_path is not None and geojson_path.exists()
    dem_available = dem_path is not None and dem_path.exists()

    return RegionStatus(
        id=meta.id,
        name=meta.name,
        description=meta.description,
        geojson_available=geojson_available,
        dem_available=dem_available,
        geojson_size_bytes=geojson_path.stat().st_size if geojson_available else None,
        dem_size_bytes=dem_path.stat().st_size if dem_available else None,
        reference_point=meta.reference_point,
        resolution_m=meta.resolution_m,
        bounds=meta.bounds,
        thumbnail=meta.thumbnail,
    )
