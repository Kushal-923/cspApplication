"""
build_matrix.py  —  Punjab Border Coverage Matrix Builder
==========================================================

Model: pure 2D Euclidean distance.
C[i][j] = 1  if  distance(radar_i, border_point_j) <= range_i
C[i][j] = 0  otherwise

Fix over previous version:
  - Inland offset now uses the correct perpendicular direction from the
    local border tangent instead of blindly adding +2000 to the X axis.

Input files (same folder):
  punjab_border.geojson

Output files (same folder):
  c_matrix.json    — 2D list  [N_radars x N_boundary_points]
  radar_meta.json  — radar names, coordinates, ranges
"""

import math
import json
import geopandas as gpd
from shapely.geometry import Point

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

BORDER_FILE   = "punjab_border.geojson"
OUTPUT_MATRIX = "c_matrix.json"
OUTPUT_META   = "radar_meta.json"

INTERVAL_M  = 50      # discretise border every 50 metres
INLAND_M    = 2000    # place radars this far inland (metres)

# Generic radar specs: [name, position along border (0.0-1.0), range in metres]
RADAR_SPECS = [
    {"name": "R1_Small",  "pos_pct": 0.10, "range_m": 5_000},
    {"name": "R2_Medium", "pos_pct": 0.30, "range_m": 10_000},
    {"name": "R3_Large",  "pos_pct": 0.50, "range_m": 15_000},
    {"name": "R4_Small",  "pos_pct": 0.70, "range_m": 5_000},
    {"name": "R5_Medium", "pos_pct": 0.90, "range_m": 10_000},
]


# ══════════════════════════════════════════════════════════════
#  STEP 1 — LOAD AND DISCRETISE THE BORDER
# ══════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("  BUILD_MATRIX.PY  —  Punjab Coverage Matrix")
print("═" * 60)

print(f"\n[1] Loading border from {BORDER_FILE} ...")
border_gdf = gpd.read_file(BORDER_FILE)

# Project to UTM Zone 43 N (EPSG:32643) so units are metres
border_gdf = border_gdf.to_crs(epsg=32643)
line = border_gdf.geometry.iloc[0]
print(f"    Border segment length : {line.length / 1000:.2f} km")

print(f"\n[2] Discretising border at {INTERVAL_M} m intervals ...")
n_pts = int(line.length // INTERVAL_M)
bpts  = [line.interpolate(i * INTERVAL_M) for i in range(n_pts + 1)]
print(f"    Boundary points       : {len(bpts)}")


# ══════════════════════════════════════════════════════════════
#  STEP 2 — PLACE RADARS WITH CORRECT PERPENDICULAR OFFSET
# ══════════════════════════════════════════════════════════════

def inland_point(line, fraction, offset_m):
    """
    Returns a Point that is offset_m metres perpendicularly inland
    from the border at position `fraction` (0.0 to 1.0).

    How it works:
      1. Find the point on the line at that fraction.
      2. Compute the local tangent direction by taking a tiny step
         forward and backward along the line.
      3. Rotate the tangent 90 degrees clockwise — this gives the
         right-hand perpendicular, which is the Indian inland side
         for a border running roughly north-south in UTM 43N.
      4. Move offset_m metres in that direction.

    Why this matters:
      The old code did  radar_x = point.x + 2000  which always shifts
      east regardless of which way the border runs at that location.
      This function always shifts truly inland no matter the orientation.
    """
    d      = fraction * line.length
    pt     = line.interpolate(d)

    # Small step to get local tangent direction
    eps    = min(1.0, line.length * 0.001)
    pt_fwd = line.interpolate(min(d + eps, line.length))
    pt_bwd = line.interpolate(max(d - eps, 0.0))

    tx = pt_fwd.x - pt_bwd.x
    ty = pt_fwd.y - pt_bwd.y
    t_len = math.hypot(tx, ty)

    if t_len == 0:
        return pt

    # Unit tangent
    tx /= t_len
    ty /= t_len

    # Right-hand perpendicular: rotate tangent 90 degrees clockwise
    nx =  ty
    ny = -tx

    return Point(pt.x + nx * offset_m, pt.y + ny * offset_m)


print(f"\n[3] Deploying {len(RADAR_SPECS)} radars ({INLAND_M}m inland) ...")
radars = []
for spec in RADAR_SPECS:
    rpt = inland_point(line, spec["pos_pct"], INLAND_M)
    radars.append({
        "name":    spec["name"],
        "x":       rpt.x,
        "y":       rpt.y,
        "range_m": spec["range_m"],
    })
    print(f"    {spec['name']:12s}  range={spec['range_m']/1000:.0f} km  "
          f"UTM=({rpt.x:.0f}, {rpt.y:.0f})")


# ══════════════════════════════════════════════════════════════
#  STEP 3 — BUILD COVERAGE MATRIX  (pure 2D Euclidean)
# ══════════════════════════════════════════════════════════════

print(f"\n[4] Building C[i][j] coverage matrix ...")
C_matrix = []

for i, radar in enumerate(radars):
    row    = []
    rx, ry = radar["x"], radar["y"]
    rng    = radar["range_m"]

    for pt in bpts:
        dist = math.hypot(pt.x - rx, pt.y - ry)
        row.append(1 if dist <= rng else 0)

    C_matrix.append(row)
    covered = sum(row)
    print(f"    {radar['name']:12s}  covers {covered:4d} / {len(bpts)} points")


# ══════════════════════════════════════════════════════════════
#  STEP 4 — EXPORT
# ══════════════════════════════════════════════════════════════

with open(OUTPUT_MATRIX, "w") as f:
    json.dump(C_matrix, f)
print(f"\n    Saved matrix   → {OUTPUT_MATRIX}")

with open(OUTPUT_META, "w") as f:
    json.dump(radars, f, indent=2)
print(f"    Saved metadata → {OUTPUT_META}")

print("\n[Done]  Run scheduler.py next.\n")
