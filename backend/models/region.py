"""
backend/models/region.py
-------------------------
Pydantic models for region data.
"""

from pydantic import BaseModel
from typing import Optional


class RegionMetadata(BaseModel):
    id: str
    name: str
    description: str
    geojson: str                  # filename relative to region dir
    dem: Optional[str] = None     # filename relative to region dir (may be absent)
    reference_point: Optional[list] = None  # [lon, lat] on the "our side" of the border
    crs_hint: Optional[str] = None          # e.g. "EPSG:32643"
    resolution_m: Optional[float] = None   # DEM pixel size in metres
    bounds: Optional[dict] = None          # {minx, miny, maxx, maxy} in WGS84
    area_km2: Optional[float] = None
    border_length_km: Optional[float] = None
    thumbnail: Optional[str] = None        # filename relative to region dir


class RegionStatus(BaseModel):
    id: str
    name: str
    description: str
    geojson_available: bool
    dem_available: bool
    geojson_size_bytes: Optional[int] = None
    dem_size_bytes: Optional[int] = None
    reference_point: Optional[list] = None
    resolution_m: Optional[float] = None
    bounds: Optional[dict] = None
    thumbnail: Optional[str] = None
