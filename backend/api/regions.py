"""
backend/api/regions.py
-----------------------
Endpoints for region discovery and metadata.
"""

from fastapi import APIRouter, HTTPException
from backend.services.region_service import list_regions, get_region, get_region_files

router = APIRouter()


@router.get("/", summary="List all available regions")
async def list_all_regions():
    """Return status information for all regions discovered in data/regions/."""
    regions = list_regions()
    return {"regions": [r.dict() for r in regions]}


@router.get("/{region_id}", summary="Get a single region")
async def get_region_detail(region_id: str):
    """Return status information for a single region."""
    region = get_region(region_id)
    if not region:
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found.")
    return region.dict()


@router.get("/{region_id}/files", summary="Get file paths for a region")
async def get_region_file_info(region_id: str):
    """Return file availability for a region (for pre-validation)."""
    try:
        files = get_region_files(region_id)
        return {
            "geojson_available": files["geojson"] is not None,
            "dem_available": files["dem"] is not None,
            "reference_point": files.get("reference_point"),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
