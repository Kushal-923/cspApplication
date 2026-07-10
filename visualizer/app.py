"""
Radar Deployment 2D Map Visualizer — Flask Backend
====================================================

Serves two data endpoints for the Leaflet frontend:
  /radars  → GeoJSON FeatureCollection of radar positions (computed from RADAR_SPECS)
  /border  → GeoJSON FeatureCollection of the Punjab border line (reprojected to WGS84)

DATA-SOURCE CONTRACT
--------------------
All radar data flows through  get_radars() → list[dict].
All border data flows through get_border_geojson() → dict.

When you're ready to swap to radar_meta.json (or any other source), replace
ONLY the body of get_radars() — nothing else in this file or the frontend
needs to change.  Same for get_border_geojson().

GEOMETRY LOGIC
--------------
Copied (not imported) from build_matrix_1_3.py Steps 1–3:
  - RADAR_SPECS, inland_point(), utm_epsg_for_lonlat()
  - Border loading / linemerge stitching
  - Constants: INLAND_M, BORDER_FILE

No DEM, no ray-marching, no matrix building — just position computation.
"""

import math
import os
import json
import functools

import geopandas as gpd
from shapely.ops import linemerge, unary_union
from shapely.geometry import Point
from pyproj import Transformer

from flask import Flask, jsonify, send_from_directory

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION (copied from build_matrix_1_3.py — cheap geometry only)
# ══════════════════════════════════════════════════════════════

# Path to the border GeoJSON, relative to the project root
# (app.py lives in visualizer/, border file is one level up)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BORDER_FILE  = os.path.join(PROJECT_ROOT, "punjab_border.geojson")

INLAND_M = 2000  # metres radar is placed inland from border

# 35 heterogeneous radars — identical to build_matrix_1_3.py
RADAR_SPECS = [
    # ── Zone 1  (0–14%) ──────────────────────────────────────
    {"name": "R01_Small",  "pos_pct": 0.01, "range_m":  5_000, "type": "Small"},
    {"name": "R02_Medium", "pos_pct": 0.04, "range_m": 10_000, "type": "Medium"},
    {"name": "R03_Large",  "pos_pct": 0.07, "range_m": 15_000, "type": "Large"},
    {"name": "R04_Medium", "pos_pct": 0.10, "range_m": 10_000, "type": "Medium"},
    {"name": "R05_Small",  "pos_pct": 0.13, "range_m":  5_000, "type": "Small"},
    # ── Zone 2  (14–28%) ─────────────────────────────────────
    {"name": "R06_Small",  "pos_pct": 0.15, "range_m":  5_000, "type": "Small"},
    {"name": "R07_Medium", "pos_pct": 0.18, "range_m": 10_000, "type": "Medium"},
    {"name": "R08_Large",  "pos_pct": 0.21, "range_m": 15_000, "type": "Large"},
    {"name": "R09_Medium", "pos_pct": 0.24, "range_m": 10_000, "type": "Medium"},
    {"name": "R10_Small",  "pos_pct": 0.27, "range_m":  5_000, "type": "Small"},
    # ── Zone 3  (28–42%) ─────────────────────────────────────
    {"name": "R11_Small",  "pos_pct": 0.29, "range_m":  5_000, "type": "Small"},
    {"name": "R12_Medium", "pos_pct": 0.32, "range_m": 10_000, "type": "Medium"},
    {"name": "R13_Large",  "pos_pct": 0.35, "range_m": 15_000, "type": "Large"},
    {"name": "R14_Medium", "pos_pct": 0.38, "range_m": 10_000, "type": "Medium"},
    {"name": "R15_Small",  "pos_pct": 0.41, "range_m":  5_000, "type": "Small"},
    # ── Zone 4  (42–56%) ─────────────────────────────────────
    {"name": "R16_Small",  "pos_pct": 0.43, "range_m":  5_000, "type": "Small"},
    {"name": "R17_Medium", "pos_pct": 0.46, "range_m": 10_000, "type": "Medium"},
    {"name": "R18_Large",  "pos_pct": 0.50, "range_m": 15_000, "type": "Large"},
    {"name": "R19_Medium", "pos_pct": 0.54, "range_m": 10_000, "type": "Medium"},
    {"name": "R20_Small",  "pos_pct": 0.57, "range_m":  5_000, "type": "Small"},
    # ── Zone 5  (56–70%) ─────────────────────────────────────
    {"name": "R21_Small",  "pos_pct": 0.59, "range_m":  5_000, "type": "Small"},
    {"name": "R22_Medium", "pos_pct": 0.62, "range_m": 10_000, "type": "Medium"},
    {"name": "R23_Large",  "pos_pct": 0.65, "range_m": 15_000, "type": "Large"},
    {"name": "R24_Medium", "pos_pct": 0.68, "range_m": 10_000, "type": "Medium"},
    {"name": "R25_Small",  "pos_pct": 0.71, "range_m":  5_000, "type": "Small"},
    # ── Zone 6  (70–84%) ─────────────────────────────────────
    {"name": "R26_Small",  "pos_pct": 0.73, "range_m":  5_000, "type": "Small"},
    {"name": "R27_Medium", "pos_pct": 0.76, "range_m": 10_000, "type": "Medium"},
    {"name": "R28_Large",  "pos_pct": 0.79, "range_m": 15_000, "type": "Large"},
    {"name": "R29_Medium", "pos_pct": 0.82, "range_m": 10_000, "type": "Medium"},
    {"name": "R30_Small",  "pos_pct": 0.85, "range_m":  5_000, "type": "Small"},
    # ── Zone 7  (84–100%) ────────────────────────────────────
    {"name": "R31_Small",  "pos_pct": 0.87, "range_m":  5_000, "type": "Small"},
    {"name": "R32_Medium", "pos_pct": 0.90, "range_m": 10_000, "type": "Medium"},
    {"name": "R33_Large",  "pos_pct": 0.93, "range_m": 15_000, "type": "Large"},
    {"name": "R34_Medium", "pos_pct": 0.96, "range_m": 10_000, "type": "Medium"},
    {"name": "R35_Small",  "pos_pct": 0.99, "range_m":  5_000, "type": "Small"},
]


# ══════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS  (copied from build_matrix_1_3.py)
# ══════════════════════════════════════════════════════════════

def utm_epsg_for_lonlat(lon, lat):
    """
    Return the EPSG code of the UTM zone containing (lon, lat).
    Copied verbatim from build_matrix_1_3.py.
    """
    zone = int((lon + 180) / 6) + 1
    zone = max(1, min(60, zone))
    return (32600 if lat >= 0 else 32700) + zone


def inland_point(line, fraction, offset_m, sign=1.0):
    """
    Offset a point along `line` at `fraction` of its length by `offset_m`
    perpendicular (inland) to the line's local tangent.
    Copied verbatim from build_matrix_1_3.py (including the sign parameter).
    """
    d      = fraction * line.length
    pt     = line.interpolate(d)

    eps    = min(1.0, line.length * 0.001)
    pt_fwd = line.interpolate(min(d + eps, line.length))
    pt_bwd = line.interpolate(max(d - eps, 0.0))

    tx = pt_fwd.x - pt_bwd.x
    ty = pt_fwd.y - pt_bwd.y
    t_len = math.hypot(tx, ty)
    if t_len == 0:
        return pt

    tx /= t_len
    ty /= t_len
    nx, ny = ty, -tx

    return Point(pt.x + sign * nx * offset_m, pt.y + sign * ny * offset_m)

# Amritsar, India — known reference point on the Indian side of the border.
# Used to determine which side of the normal is "inland" (India) on a
# per-radar basis, exactly as build_matrix_1_3.py does in STEP 2.5 + 3.
REFERENCE_INDIA_LONLAT = (74.8723, 31.6340)


def _load_border():
    """
    Load and merge the border GeoJSON into a single UTM LineString.
    Returns (line, target_epsg, ref_pt).
    Replicates build_matrix_1_3.py Steps 1 + 2.5 logic.
    """
    border_gdf = gpd.read_file(BORDER_FILE)

    # Auto-detect UTM zone from the border's own centroid
    border_wgs84 = border_gdf.to_crs(epsg=4326).geometry
    merged_wgs84 = (
        border_wgs84.union_all()
        if hasattr(border_wgs84, "union_all")
        else border_wgs84.unary_union
    )
    centroid = merged_wgs84.centroid
    target_epsg = utm_epsg_for_lonlat(centroid.x, centroid.y)

    # Reproject to UTM and merge all segments into one LineString
    border_gdf = border_gdf.to_crs(epsg=target_epsg)
    line = linemerge(unary_union(border_gdf.geometry))

    if line.geom_type != "LineString":
        raise RuntimeError(
            f"linemerge produced {line.geom_type} — segments may have gaps. "
            "Inspect the GeoJSON for discontinuities."
        )

    # Project the Amritsar reference point into UTM for per-radar sign checks
    ref_gdf = gpd.GeoDataFrame(
        geometry=[Point(REFERENCE_INDIA_LONLAT)], crs="EPSG:4326"
    ).to_crs(epsg=target_epsg)
    ref_pt = ref_gdf.geometry.iloc[0]

    return line, target_epsg, ref_pt


# Cache the border computation — it's the same every time
@functools.lru_cache(maxsize=1)
def _cached_border():
    return _load_border()


# ══════════════════════════════════════════════════════════════
#  DATA-SOURCE FUNCTIONS  (the swap points)
# ══════════════════════════════════════════════════════════════

def get_radars():
    """
    Compute radar positions from RADAR_SPECS and return as a list of dicts.

    Schema per radar:
        name               : str
        lat                : float   (WGS84)
        lon                : float   (WGS84)
        range_m            : int
        type               : str     ("Large" | "Medium" | "Small")
        frequency          : None    (not yet available)
        physical_conditions: None    (not yet available)

    The sign for each radar's inland offset is determined individually
    by checking the dot product of the local normal with the direction
    toward the Amritsar reference point. This handles border curves
    where the normal direction flips.

    ── SWAP POINT ──
    To read from radar_meta.json instead, replace this function body.
    Nothing else in the app needs to change.
    """
    line, target_epsg, ref_pt = _cached_border()

    # UTM → WGS84 transformer
    transformer = Transformer.from_crs(
        f"EPSG:{target_epsg}", "EPSG:4326", always_xy=True
    )

    radars = []
    for spec in RADAR_SPECS:
        # Compute the local normal at this position along the border
        d      = spec["pos_pct"] * line.length
        pt     = line.interpolate(d)
        eps    = min(1.0, line.length * 0.001)
        pt_fwd = line.interpolate(min(d + eps, line.length))
        pt_bwd = line.interpolate(max(d - eps, 0.0))
        tx = pt_fwd.x - pt_bwd.x
        ty = pt_fwd.y - pt_bwd.y
        t_len = math.hypot(tx, ty)
        if t_len > 0:
            tx /= t_len; ty /= t_len
        nx, ny = ty, -tx

        # Dot product with direction toward Amritsar determines the sign
        vx = ref_pt.x - pt.x
        vy = ref_pt.y - pt.y
        dot = nx * vx + ny * vy
        sign = 1.0 if dot > 0 else -1.0

        rpt = inland_point(line, spec["pos_pct"], INLAND_M, sign=sign)

        # Convert UTM (x, y) → lon, lat
        lon, lat = transformer.transform(rpt.x, rpt.y)

        radars.append({
            "name":                spec["name"],
            "lat":                 round(lat, 6),
            "lon":                 round(lon, 6),
            "range_m":             spec["range_m"],
            "type":                spec["type"],
            "frequency":           None,
            "physical_conditions": None,
        })

    return radars


def get_border_geojson():
    """
    Load the border GeoJSON, reproject all geometries to WGS84 (EPSG:4326),
    and return as a GeoJSON dict (FeatureCollection).

    ── SWAP POINT ──
    To serve a different border source, replace this function body.
    """
    border_gdf = gpd.read_file(BORDER_FILE)
    border_wgs84 = border_gdf.to_crs(epsg=4326)
    return json.loads(border_wgs84.to_json())


# ══════════════════════════════════════════════════════════════
#  FLASK APP
# ══════════════════════════════════════════════════════════════

app = Flask(
    __name__,
    static_folder="static",
    static_url_path="/static",
)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/radars")
def radars_endpoint():
    """Return radar data as a GeoJSON FeatureCollection of Points."""
    radars = get_radars()
    features = []
    for r in radars:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [r["lon"], r["lat"]],
            },
            "properties": {
                "name":                r["name"],
                "lat":                 r["lat"],
                "lon":                 r["lon"],
                "range_m":             r["range_m"],
                "type":                r["type"],
                "frequency":           r["frequency"],
                "physical_conditions": r["physical_conditions"],
            },
        })
    return jsonify({"type": "FeatureCollection", "features": features})


@app.route("/border")
def border_endpoint():
    """Return the border line as a GeoJSON FeatureCollection."""
    return jsonify(get_border_geojson())


@app.route("/schedule")
def schedule_endpoint():
    """
    Return the schedule from schedule.json (written by scheduler_1_3.py).

    Response on success:
        { N, M, phase, status, schedule: [[int]] }  — as-is from the file

    Response when file is missing or unreadable:
        { error: "...", missing: true }              — frontend shows a prompt

    Radar index i in schedule[i] corresponds to the radar at index i in
    RADAR_SPECS (same ordering as c_matrix.json). The frontend matches by
    positional index — no name string lookup needed.
    """
    schedule_path = os.path.join(PROJECT_ROOT, "schedule.json")
    try:
        with open(schedule_path, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({
            "error": "schedule.json not found — run scheduler_1_3.py to generate a schedule",
            "missing": True,
        }), 404
    except Exception as e:
        return jsonify({
            "error": f"Could not read schedule.json: {e}",
            "missing": True,
        }), 500



if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Radar Deployment 2D Visualizer")
    print("=" * 50)
    print("  Preloading border + radar data ...")


    # Eagerly load so first page-load isn't slow
    _cached_border()
    radars = get_radars()
    print(f"  {len(radars)} radars computed")
    print(f"  Open  http://localhost:5000  in your browser")
    print("=" * 50 + "\n")

    app.run(host="127.0.0.1", port=5000, debug=False)
