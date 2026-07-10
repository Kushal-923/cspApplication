"""
build_matrix_1.2.py  —  Punjab Border Coverage Matrix Builder
==============================================================

Key fix over v1.1:
  v1.1 did:  line = border_gdf.geometry.iloc[0]
             → grabbed only feature[0] = 26.5 km (first OSM way)

  v1.2 does: line = linemerge(unary_union(all 34 features))
             → stitches all OSM ways into one continuous line = ~510 km

Why the GeoJSON has 34 features:
  OpenStreetMap stores borders as individual "way" objects, each covering
  one contiguous segment of the border.  The file contains all 34 ways that
  make up the India-Pakistan border as tagged in OSM.  linemerge() joins
  them end-to-end wherever endpoints match, producing a single LineString.

Resolution guidance (INTERVAL_M):
  50 m  → ~10 200 boundary pts  — very fine, solver will be slow
  100 m → ~5 100 boundary pts   — good balance for research
  250 m → ~2 000 boundary pts   — fast solver, acceptable for initial runs
  500 m → ~1 000 boundary pts   — prototype / debug only

Radar fleet (35 radars, heterogeneous):
  7 × Large  (range 15 km)  — one per ~70 km stretch
  14 × Medium (range 10 km) — two per ~70 km stretch
  14 × Small  (range  5 km) — two per ~70 km stretch
  All placed 2000 m inland using the correct perpendicular tangent method.

Coverage model (v1.2 upgrade — full reference math):
  C[i][j] = 1  iff  ALL three gates pass:

  Gate A — 3D slant-range:
    d_ij = sqrt(Δx²+Δy²+Δz²)  ≤  R_i
    (true slant distance, not just horizontal)

  Gate B — curvature-corrected LOS ray test:
    Ray is sampled at S = ceil(d_horiz / 30 m) steps (one per DEM pixel).
    At each step s ∈ {1/S, 2/S, …, 1}:
      ray_z(s)     = z_i + s·(z_j − z_i)          linear height
      h_bulge(s)   = s·(1−s)·d_horiz²/(2·R_eff)   earth-curvature lift
      terrain_z(s) = DEM(x_i + s·Δx, y_i + s·Δy)  bilinear DEM sample
    Blocked if ray_z(s) ≤ terrain_z(s) + h_bulge(s) at any step.

  Gate C — field-of-view / elevation angle:
    θ_ij = arctan(Δz / d_horiz)   (negative = looking downward)
    Must satisfy  FOV_ELEV_MIN_DEG ≤ θ_ij ≤ FOV_ELEV_MAX_DEG.

Input:   punjab_border.geojson  +  punjab_dem.tif
Output:  c_matrix.json    (N_radars × K_boundary_points binary matrix)
         radar_meta.json  (radar names, UTM coords, ranges)
"""

import math
import json
import geopandas as gpd
import rasterio
from pyproj import Transformer
from shapely.ops import linemerge, unary_union
from shapely.geometry import Point

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

BORDER_FILE   = "punjab_border.geojson"
DEM_PATH      = "punjab_dem.tif"          # GeoTIFF DEM in UTM 43N (metres)
OUTPUT_MATRIX = "c_matrix.json"
OUTPUT_META   = "radar_meta.json"

# 3-D LOS physics constants
ANTENNA_HEIGHT_M  = 15.0          # radar antenna mast above ground (metres)
R_EFF             = 8_494_667.0   # effective Earth radius for k=4/3 atmosphere (metres)

# Full raycast parameters (Gate B)
DEM_RESOLUTION_M  = 30.0   # GLO-30 pixel size — one terrain sample per DEM pixel

# Field-of-view elevation angle limits (Gate C)
# θ = arctan(Δz / d_horiz): negative = looking below horizontal
FOV_ELEV_MIN_DEG  = -2.0   # e.g. allow slight downward look (avoids ground clutter)
FOV_ELEV_MAX_DEG  = 30.0   # max upward elevation angle (mast-mounted radar)

# Discretisation resolution.
# Recommended: start at 250 m for testing, move to 100 m for final runs.
INTERVAL_M  = 250     # metres between consecutive boundary points
INLAND_M    = 2000    # metres radar is placed inland from border

# 35 heterogeneous radars spread evenly along the border.
# pos_pct is fractional position along the merged line (0.0 = start, 1.0 = end).
# Pattern per zone: Large at centre, Medium either side, Small at edges.
RADAR_SPECS = [
    # ── Zone 1  (0–14%) ──────────────────────────────────────
    {"name": "R01_Small",  "pos_pct": 0.01, "range_m":  5_000},
    {"name": "R02_Medium", "pos_pct": 0.04, "range_m": 10_000},
    {"name": "R03_Large",  "pos_pct": 0.07, "range_m": 15_000},
    {"name": "R04_Medium", "pos_pct": 0.10, "range_m": 10_000},
    {"name": "R05_Small",  "pos_pct": 0.13, "range_m":  5_000},
    # ── Zone 2  (14–28%) ─────────────────────────────────────
    {"name": "R06_Small",  "pos_pct": 0.15, "range_m":  5_000},
    {"name": "R07_Medium", "pos_pct": 0.18, "range_m": 10_000},
    {"name": "R08_Large",  "pos_pct": 0.21, "range_m": 15_000},
    {"name": "R09_Medium", "pos_pct": 0.24, "range_m": 10_000},
    {"name": "R10_Small",  "pos_pct": 0.27, "range_m":  5_000},
    # ── Zone 3  (28–42%) ─────────────────────────────────────
    {"name": "R11_Small",  "pos_pct": 0.29, "range_m":  5_000},
    {"name": "R12_Medium", "pos_pct": 0.32, "range_m": 10_000},
    {"name": "R13_Large",  "pos_pct": 0.35, "range_m": 15_000},
    {"name": "R14_Medium", "pos_pct": 0.38, "range_m": 10_000},
    {"name": "R15_Small",  "pos_pct": 0.41, "range_m":  5_000},
    # ── Zone 4  (42–56%) ─────────────────────────────────────
    {"name": "R16_Small",  "pos_pct": 0.43, "range_m":  5_000},
    {"name": "R17_Medium", "pos_pct": 0.46, "range_m": 10_000},
    {"name": "R18_Large",  "pos_pct": 0.50, "range_m": 15_000},
    {"name": "R19_Medium", "pos_pct": 0.54, "range_m": 10_000},
    {"name": "R20_Small",  "pos_pct": 0.57, "range_m":  5_000},
    # ── Zone 5  (56–70%) ─────────────────────────────────────
    {"name": "R21_Small",  "pos_pct": 0.59, "range_m":  5_000},
    {"name": "R22_Medium", "pos_pct": 0.62, "range_m": 10_000},
    {"name": "R23_Large",  "pos_pct": 0.65, "range_m": 15_000},
    {"name": "R24_Medium", "pos_pct": 0.68, "range_m": 10_000},
    {"name": "R25_Small",  "pos_pct": 0.71, "range_m":  5_000},
    # ── Zone 6  (70–84%) ─────────────────────────────────────
    {"name": "R26_Small",  "pos_pct": 0.73, "range_m":  5_000},
    {"name": "R27_Medium", "pos_pct": 0.76, "range_m": 10_000},
    {"name": "R28_Large",  "pos_pct": 0.79, "range_m": 15_000},
    {"name": "R29_Medium", "pos_pct": 0.82, "range_m": 10_000},
    {"name": "R30_Small",  "pos_pct": 0.85, "range_m":  5_000},
    # ── Zone 7  (84–100%) ────────────────────────────────────
    {"name": "R31_Small",  "pos_pct": 0.87, "range_m":  5_000},
    {"name": "R32_Medium", "pos_pct": 0.90, "range_m": 10_000},
    {"name": "R33_Large",  "pos_pct": 0.93, "range_m": 15_000},
    {"name": "R34_Medium", "pos_pct": 0.96, "range_m": 10_000},
    {"name": "R35_Small",  "pos_pct": 0.99, "range_m":  5_000},
]


# ══════════════════════════════════════════════════════════════
#  STEP 1 — LOAD AND STITCH ALL 34 OSM SEGMENTS
# ══════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("  BUILD_MATRIX v1.2  —  Full Punjab Border (~510 km)")
print("═" * 60)

print(f"\n[1] Loading {BORDER_FILE} ...")
border_gdf = gpd.read_file(BORDER_FILE)
print(f"    Raw features in GeoJSON : {len(border_gdf)}")

# Project to UTM Zone 43N (EPSG:32643) — units become metres
border_gdf = border_gdf.to_crs(epsg=32643)

# ── THE KEY FIX ───────────────────────────────────────────────
# v1.1: line = border_gdf.geometry.iloc[0]   ← only 26.5 km!
# v1.2: merge ALL 34 features into one LineString
line = linemerge(unary_union(border_gdf.geometry))

if line.geom_type != "LineString":
    # Should not happen with this GeoJSON, but guard anyway
    raise RuntimeError(
        f"linemerge produced {line.geom_type} — segments may have gaps. "
        "Inspect the GeoJSON for discontinuities."
    )

print(f"    Features merged         : {len(border_gdf)} → 1 LineString")
print(f"    Total border length     : {line.length / 1000:.1f} km")


# ══════════════════════════════════════════════════════════════
#  STEP 2 — DISCRETISE INTO BOUNDARY POINTS
# ══════════════════════════════════════════════════════════════

print(f"\n[2] Discretising at {INTERVAL_M} m intervals ...")
n_pts = int(line.length // INTERVAL_M)
bpts  = [line.interpolate(i * INTERVAL_M) for i in range(n_pts + 1)]
print(f"    K = {len(bpts)} boundary points")
print(f"    Coverage resolution     : one point per {INTERVAL_M} m")


# ══════════════════════════════════════════════════════════════
#  STEP 3 — DEPLOY RADARS USING PERPENDICULAR TANGENT OFFSET
# ══════════════════════════════════════════════════════════════

def inland_point(line, fraction, offset_m):
    """
    Place a radar offset_m metres perpendicularly inland from the border.

    Method:
      1. Locate the point on the line at the given fractional position.
      2. Compute the local tangent by sampling a tiny step forward and back.
      3. Rotate the tangent 90° clockwise → right-hand perpendicular,
         which points inland (east) for a border running roughly N→S in
         UTM 43N.
      4. Displace the border point by offset_m in that direction.

    This is correct regardless of local border orientation — unlike the
    old approach of blindly adding a fixed X offset.
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
        return pt          # degenerate point — return on-border position

    tx /= t_len
    ty /= t_len

    # 90° clockwise rotation of unit tangent → inland normal
    nx =  ty
    ny = -tx

    return Point(pt.x + nx * offset_m, pt.y + ny * offset_m)


print(f"\n[3] Deploying {len(RADAR_SPECS)} radars at {INLAND_M} m inland ...")
radars = []
for spec in RADAR_SPECS:
    rpt = inland_point(line, spec["pos_pct"], INLAND_M)
    radars.append({
        "name":    spec["name"],
        "x":       rpt.x,
        "y":       rpt.y,
        "range_m": spec["range_m"],
    })

# Summary by type
small  = sum(1 for r in radars if "Small"  in r["name"])
medium = sum(1 for r in radars if "Medium" in r["name"])
large  = sum(1 for r in radars if "Large"  in r["name"])
print(f"    {large} × Large (15 km)  |  {medium} × Medium (10 km)  |  {small} × Small (5 km)")
print(f"    Total radars N = {len(radars)}")


# ══════════════════════════════════════════════════════════════
#  DEM HELPER — sample terrain elevation from GeoTIFF
# ══════════════════════════════════════════════════════════════

print(f"\n[3.5] Opening DEM: {DEM_PATH} ...")
_dem_dataset = rasterio.open(DEM_PATH)

# The DEM may be in a different CRS (e.g. EPSG:4326 for GLO-30).
# Build a transformer from UTM 43N → DEM native CRS so every
# elevation lookup gets coordinates the raster actually understands.
_dem_epsg = _dem_dataset.crs.to_epsg()
if _dem_epsg != 32643:
    _utm_to_dem = Transformer.from_crs(
        "EPSG:32643", _dem_dataset.crs, always_xy=True
    )
else:
    _utm_to_dem = None   # DEM is already in UTM 43N — no transform needed

def get_elevation(x: float, y: float) -> float:
    """
    Sample the DEM GeoTIFF at a UTM 43N coordinate (x=easting, y=northing)
    and return the terrain height in metres.

    Coordinates are automatically converted to the DEM's native CRS before
    the pixel lookup, so this works whether the GeoTIFF is in EPSG:4326
    (GLO-30 default) or already in UTM 43N.

    If the coordinate falls outside the raster extent the function returns
    0.0 so the LOS check degrades gracefully rather than crashing.
    """
    if _utm_to_dem is not None:
        x, y = _utm_to_dem.transform(x, y)   # UTM → DEM CRS (e.g. lon/lat)
    try:
        row, col = _dem_dataset.index(x, y)
        # rasterio.index can return out-of-bounds indices for exterior points
        if (row < 0 or col < 0
                or row >= _dem_dataset.height
                or col >= _dem_dataset.width):
            return 0.0
        window = rasterio.windows.Window(col, row, 1, 1)
        data   = _dem_dataset.read(1, window=window)
        value  = float(data[0, 0])
        # Treat nodata as flat ground
        nd = _dem_dataset.nodata
        if nd is not None and value == nd:
            return 0.0
        return value
    except Exception:
        return 0.0

print(f"    DEM CRS  : {_dem_dataset.crs}")
print(f"    DEM size : {_dem_dataset.width} × {_dem_dataset.height} px")
print(f"    DEM res  : {_dem_dataset.res[0] * 111_000:.0f} m/px  "
      f"(native unit: {_dem_dataset.res[0]:.6f}°)")
if _utm_to_dem is not None:
    print(f"    Coord transform: UTM 43N → EPSG:{_dem_epsg} (auto)")


# ══════════════════════════════════════════════════════════════
#  STEP 4 — BUILD COVERAGE MATRIX  C[i][j]  (3D LOS model)
# ══════════════════════════════════════════════════════════════

print(f"\n[4] Building {len(radars)} × {len(bpts)} coverage matrix (3D LOS + DEM) ...")

# Pre-compute radar 3-D heights (DEM + antenna mast)
radar_z = [
    get_elevation(r["x"], r["y"]) + ANTENNA_HEIGHT_M
    for r in radars
]

# Pre-compute boundary point ground elevations
bpt_z = [get_elevation(pt.x, pt.y) for pt in bpts]

C_matrix = []
total_coverage_cells = 0

for i, radar in enumerate(radars):
    rx, ry = radar["x"], radar["y"]
    rng    = radar["range_m"]
    zi     = radar_z[i]          # radar antenna height (m ASL)
    row    = []

    for j, pt in enumerate(bpts):
        dx      = pt.x - rx
        dy      = pt.y - ry
        d_horiz = math.hypot(dx, dy)
        zj      = bpt_z[j]       # boundary point ground elevation (m ASL)
        dz      = zj - zi        # signed vertical offset (negative = target below radar)

        # ── Gate A: 3D slant-range check ──────────────────────────
        # Use true slant distance, not just horizontal range.
        d_3d = math.sqrt(d_horiz * d_horiz + dz * dz)
        if d_3d > rng:
            row.append(0)
            continue

        # ── Gate B: full S-step curvature-corrected LOS ray ───────
        # S = one sample per DEM pixel (~30 m), so no ridge is skipped.
        # s ∈ {1/S, 2/S, …, 1}  — endpoint (target) is also checked.
        S   = max(1, math.ceil(d_horiz / DEM_RESOLUTION_M))
        los = True
        for k in range(1, S + 1):
            s          = k / S
            ray_x      = rx + s * dx
            ray_y      = ry + s * dy
            ray_z      = zi + s * dz                              # linear height
            h_bulge    = s * (1.0 - s) * d_horiz**2 / (2.0 * R_EFF)  # curvature bulge
            terrain_z  = get_elevation(ray_x, ray_y)
            if ray_z < terrain_z + h_bulge:    # ray goes below terrain
                los = False
                break
        if not los:
            row.append(0)
            continue

        # ── Gate C: field-of-view / elevation angle check ─────────
        # θ_ij = arctan(Δz / d_horiz); negative means looking downward.
        if d_horiz > 0:
            theta_deg = math.degrees(math.atan2(dz, d_horiz))
        else:
            theta_deg = 90.0 if dz > 0 else -90.0
        if not (FOV_ELEV_MIN_DEG <= theta_deg <= FOV_ELEV_MAX_DEG):
            row.append(0)
            continue

        row.append(1)

    C_matrix.append(row)
    covered = sum(row)
    total_coverage_cells += covered
    pct = 100 * covered / len(bpts)
    print(f"    {radar['name']:12s}  range={rng/1000:2.0f} km  "
          f"covers {covered:4d}/{len(bpts)} pts  ({pct:5.1f}%)")

# Close the DEM dataset now that we're done sampling
_dem_dataset.close()

# Sanity check: how many boundary points have zero coverage at all?
uncovered = sum(
    1 for j in range(len(bpts))
    if all(C_matrix[i][j] == 0 for i in range(len(radars)))
)
print(f"\n    Total coverage cells    : {total_coverage_cells}")
print(f"    Permanently uncoverable : {uncovered} / {len(bpts)} points "
      f"({100*uncovered/len(bpts):.1f}%)")
if uncovered > 0:
    print("    ⚠  Some points cannot be covered by ANY radar.")
    print("       Scheduler will route into Phase 1 (minimise void).")
else:
    print("    ✓  Every point is reachable by at least one radar.")


# ══════════════════════════════════════════════════════════════
#  STEP 5 — EXPORT
# ══════════════════════════════════════════════════════════════

print(f"\n[5] Exporting ...")
with open(OUTPUT_MATRIX, "w") as f:
    json.dump(C_matrix, f)
print(f"    Saved matrix   → {OUTPUT_MATRIX}  "
      f"({len(radars)} radars × {len(bpts)} points)")

with open(OUTPUT_META, "w") as f:
    json.dump(radars, f, indent=2)
print(f"    Saved metadata → {OUTPUT_META}")

print(f"\n[Done]  Run scheduler_1.2.py next.\n")
