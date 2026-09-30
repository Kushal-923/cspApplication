"""
build_matrix_1.3.py  —  Border Coverage Matrix Builder (3D, terrain-aware)
===================================================================================

What changed vs v1.2 (2D Euclidean):
  v1.2:  C_ij = 1  if  euclidean_2d(radar_i, point_j) <= range_i
  v1.3:  C_ij = 1  if  d_ij <= range_i
                  AND LOS(i,j) = 1     (ray-marched, curvature-corrected)
                  AND FOV(i,j) = 1     (elevation angle window, optional)

Everything below implements the four blocks from the math reference doc:
  1. Earth curvature bulge term
  2. LOS ray test (curvature-corrected)
  3. FOV / elevation angle check
  4. Final coverage condition C_ij

scheduler.py is untouched — it only ever reads c_matrix.json, and that
contract (N_radars x K_boundary_points binary matrix) is preserved exactly.

Corrections made to the v1.3 math draft during implementation (see chat
for full derivation):
  - d_horiz = 0 edge case (radar directly above/below a point) is handled
    explicitly — the original formula would silently pass with zero ray
    samples instead of raising or defaulting sensibly.
  - LOS_EPS tolerance added at the terrain comparison to avoid float/DEM
    interpolation noise flipping a true-clear ray to "blocked" right at
    s=1, where ray_z(1) and terrain_z(1) are mathematically equal.
  - H_TARGET added (default 0 m, i.e. worst-case ground-level target).
    z_j was DEM(x_j,y_j) with no target height in the draft; that's kept
    as the default but is now a named, changeable config value instead
    of an implicit assumption baked into the geometry.

v1.3.1 patch (post-review fixes, both integrated):
  - LOS ray sampling density: was S = ceil(d_horiz/30), which under-samples
    at exactly DEM pixel resolution (aliasing risk — a ridge could fall
    between two samples) and degenerates to a single sample for short
    hops. Now S = max(4, ceil(d_horiz / 15)) — double-resolution sampling
    with a sane floor.
  - DEMReader.elevations() now checks that sample coordinates fall inside
    the raster's actual footprint BEFORE clipping to it, and raises
    RuntimeError if not. Previously, a radar or point outside DEM
    coverage would silently get the DEM's edge-pixel elevation instead
    of an error — a different failure mode than the NODATA check, which
    only catches void pixels *inside* the raster.

Not yet resolved (flagged, not silently decided):
  - FOV bounds (theta_min/theta_max) and azimuth half-widths are no longer
    hardcoded in this file — they are loaded per-type from
    radar_type_config.json (see STEP 3/STEP 5 below), a file produced by
    optimize_placement.py from user input. ENABLE_FOV_CHECK /
    ENABLE_AZIMUTH_CHECK below remain this file's own toggles for whether
    those loaded bounds are actually enforced when building c_matrix.json.
  - Added an AZIMUTH (horizontal) FOV check — in_fov() only ever tested
    elevation, so every radar was implicitly 360-deg omnidirectional in
    the horizontal plane. in_azimuth() adds a directional sector test
    against each radar's boresight_deg (loaded from
    radar_meta_optimized.json; originally derived by
    optimize_placement.py's inland_point(), facing back toward the border
    segment the radar was offset from). Independent toggle:
    ENABLE_AZIMUTH_CHECK, same as ENABLE_FOV_CHECK.

Input:   punjab_border.geojson
         <DEM_FILE>                — GLO-30 (or similar) DEM raster, any CRS
                                      (reprojected automatically to the
                                      auto-detected UTM zone)
         radar_meta_optimized.json — radar placement (name/x/y/range_m/type/
                                      boresight_deg/ground_z/h_ant/z), produced
                                      by optimize_placement.py. This script no
                                      longer computes placement itself; it is a
                                      pure "given a radar list + border + DEM,
                                      produce c_matrix.json" step.
         radar_type_config.json    — per-type FOV/azimuth bounds (theta_min,
                                      theta_max, az_halfwidth), keyed by type
                                      name, also produced by
                                      optimize_placement.py.
Output:  c_matrix.json    (N_radars x K_boundary_points binary matrix)
         radar_meta.json  (pass-through re-export of the loaded radar list,
                            same schema as radar_meta_optimized.json — kept
                             under this filename so scheduler.py and
                            generate_czml.py need no changes)

Cross-file invariant introduced by consuming external placement (see also
CLAUDE.md):
  - Every radar's "type" string in radar_meta_optimized.json MUST exist as a
    key in radar_type_config.json. Both files are written together by a
    single optimize_placement.py run, so this should hold by construction —
    but if radar_meta_optimized.json is hand-edited, or the two files come
    from different optimize_placement.py runs, this script fails fast with a
    RuntimeError naming the offending radar and type rather than silently
    skipping FOV/azimuth checks for it.
  - ENABLE_FOV_CHECK / ENABLE_AZIMUTH_CHECK below should be kept consistent
    with the same-named toggles in optimize_placement.py. optimize_placement.py
    uses its own copies of these toggles to decide what coverage semantics to
    OPTIMIZE the placement for; this script uses them to decide what coverage
    semantics to actually ENCODE into c_matrix.json. If the two scripts
    disagree, the radar positions could be well-suited to a coverage
    definition this script no longer computes the same way.
"""

import math
import json
import sys
import numpy as np
import geopandas as gpd
from shapely.ops import linemerge, unary_union

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import map_coordinates

# Console output below uses box-drawing/checkmark characters -- force
# UTF-8 on stdout so this doesn't crash under a plain Windows console
# (cp1252), which can't encode them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

BORDER_FILE   = "punjab_border.geojson"
DEM_FILE      = "punjab_dem.tif"     # GLO-30 or similar; any CRS, any tiling
OUTPUT_MATRIX = "c_matrix.json"
OUTPUT_META   = "radar_meta.json"

# Radar placement + per-type config, produced by optimize_placement.py.
# This script no longer computes placement itself -- it loads a
# fully-computed radar list. Filenames match optimize_placement.py's own
# OUTPUT_PLACEMENT / OUTPUT_TYPE_CONFIG constants exactly -- hand-edit
# both together if you rename either output there.
INPUT_PLACEMENT   = "radar_meta_optimized.json"
INPUT_TYPE_CONFIG = "radar_type_config.json"

# UTM zone is now AUTO-DETECTED from BORDER_FILE's own location (see
# utm_epsg_for_lonlat() + STEP 1 below) instead of hardcoded. 32643 (UTM
# 43N) was only correct because its central meridian, 75°E, happens to
# run through Punjab. Point BORDER_FILE at a geojson for a different
# state and a hardcoded 32643 would silently reproject it into the wrong
# zone — no crash, just quietly distorted distances that get worse the
# further the new AOI is from 75°E. TARGET_EPSG is computed once border_gdf
# is loaded, and used everywhere downstream (radar placement, DEM reproj).
TARGET_EPSG = None   # placeholder; set for real in STEP 1

INTERVAL_M  = 250     # metres between consecutive boundary points

# Target height above ground (z_j = DEM(x_j,y_j) + H_TARGET).
# 0.0 = worst case, ground-hugging target. Bump to ~1.8m for "person
# standing", ~2.5m for "vehicle", if that matches your threat model.
H_TARGET = 0.0

# ── Earth curvature ──────────────────────────────────────────
#
# Physical constants for terrain LOS Earth-curvature correction.
#
# EARTH_RADIUS_M (R_E)
#   The geometric mean radius of the Earth, ~6,371 km.  This is the
#   physical radius of the sphere that best approximates the real geoid.
#   It is NOT what we use directly in LOS calculations because the
#   atmosphere bends radio rays downward, which effectively lets them
#   "see over" terrain slightly further than pure geometry would allow.
#
# REFRACTION_FACTOR (K_REFRACTION)
#   The atmospheric refraction factor k.  Under standard tropospheric
#   conditions (temperature/pressure/humidity lapse rates per ITU-R P.834),
#   radio rays travel in arcs as if the Earth were flat but had a radius
#   of k × R_E.  The conventional value k = 4/3 (≈ 1.333) is the
#   standard engineering approximation for temperate climates; k = 1.0
#   means no atmospheric bending (geometric optics only); k > 1 means
#   rays bend toward Earth (increased effective horizon range).
#
# EFFECTIVE_EARTH_RADIUS (R_EFF)
#   R_EFF = K_REFRACTION × R_E  — the "effective Earth radius" used in
#   h_bulge() to compute the terrain-curvature correction applied to
#   every LOS ray sample.  Change ONLY this derived constant affects the
#   LOS geometry; it is intentionally kept as an explicit product so the
#   physical (R_E) and atmospheric (K_REFRACTION) contributions remain
#   separately identifiable.
#
# To change the model:
#   - Different atmospheric condition: adjust K_REFRACTION only.
#   - Different planet / ellipsoid: adjust R_E only.
#   - Default (k=4/3, R_E=6371 km) reproduces standard radio-horizon
#     practice and matches optimize_placement.py's identical block.
#
ENABLE_EARTH_CURVATURE = True
EARTH_RADIUS_M  = 6_371_000.0   # geometric mean Earth radius [m]
REFRACTION_FACTOR = 4.0 / 3.0   # standard tropospheric refraction factor k
# Effective Earth radius used in all LOS bulge calculations:
#   R_EFF = k × R_E ≈ 8,494,667 m
# Increasing k makes the horizon farther (more coverage per LOS ray);
# setting k=1.0 reverts to pure geometric line-of-sight.
R_E   = EARTH_RADIUS_M          # backward-compat alias (used nowhere else internally)
K_REFRACTION = REFRACTION_FACTOR # backward-compat alias
R_EFF = REFRACTION_FACTOR * EARTH_RADIUS_M  # ← used by h_bulge()

# ── LOS ray march ────────────────────────────────────────────
DEM_PIXEL_M = 30.0            # GLO-30 resolution -> one check per pixel
LOS_EPS_M   = 0.5             # tolerance at the terrain comparison

# ── FOV / elevation window ────────────────────────────────────
# Bounds themselves are no longer hardcoded here -- loaded per-type from
# radar_type_config.json in STEP 3, looked up per-radar in STEP 5.
ENABLE_FOV_CHECK = True

# ── FOV / azimuth window (horizontal field of view) ───────────
# in_fov() above only ever checked elevation (vertical angle). It had
# NO azimuth/bearing term at all, so every radar was implicitly treated
# as 360 deg omnidirectional in the horizontal plane — it could "see"
# radially inward, sideways, behind itself, etc., as long as elevation
# + range + LOS passed. This block adds a horizontal sector check on
# top of that, independent of ENABLE_FOV_CHECK (elevation) so either
# axis can be toggled on its own while values are being tuned.
#
# BORESIGHT_DEG per radar = the direction the radar "faces". Computed once
# by optimize_placement.py's inland_point() at placement time and loaded
# here, already present as radar["boresight_deg"], in STEP 3.
#
# Angle convention: standard math bearing, atan2(dy, dx) in degrees,
# 0 deg = +x (East), 90 deg = +y (North), increasing counter-clockwise.
# This matches the convention used in in_azimuth() below — do not mix
# with compass bearing (0=N, clockwise) without converting both ends.
#
# Halfwidth bounds themselves are no longer hardcoded here -- loaded
# per-type from radar_type_config.json in STEP 3, looked up per-radar in
# STEP 5, same as the elevation FOV bounds above.
ENABLE_AZIMUTH_CHECK = True


# ══════════════════════════════════════════════════════════════
#  DEM READER
#  Wraps a GDAL raster and exposes vectorised bilinear elevation
#  lookups at arbitrary continuous UTM coordinates. This is the
#  "DEM(x,y)" black box the math doc refers to.
# ══════════════════════════════════════════════════════════════

class DEMReader:
    def __init__(self, path, target_epsg):
        # No default here on purpose: TARGET_EPSG is only known after
        # STEP 1 auto-detects it from BORDER_FILE's own location, so a
        # default captured at class-definition time would silently be
        # the stale placeholder (None) instead of the real zone.
        src = rasterio.open(path)

        if src.crs is None:
            raise RuntimeError(
                f"{path} has no CRS defined — cannot safely reproject. "
                "Set the CRS on the source raster first."
            )

        # Both branches below actually read pixel data (a full-band read,
        # or a reprojection warp that reads tiles progressively) -- either
        # can fail with a low-level GDAL/rasterio error if the file is
        # corrupted or truncated (e.g. an interrupted download that got
        # saved as if it completed). Both failure shapes share the common
        # base class rasterio.errors.RasterioError, so one except clause
        # catches both and turns a cryptic GDAL traceback into a clear,
        # actionable message instead.
        try:
            if src.crs.to_epsg() == target_epsg:
                # Already in the right frame — use as-is.
                self.array = src.read(1).astype(np.float64)
                self.transform = src.transform
                self.nodata = src.nodata
            else:
                # Reproject the whole raster into EPSG:32643 once, up front,
                # rather than reprojecting coordinates per-sample. Cheaper
                # and keeps the bilinear math in one consistent grid.
                dst_transform, width, height = calculate_default_transform(
                    src.crs, f"EPSG:{target_epsg}", src.width, src.height, *src.bounds
                )
                dst_array = np.empty((height, width), dtype=np.float64)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst_array,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=f"EPSG:{target_epsg}",
                    resampling=Resampling.bilinear,
                )
                self.array = dst_array
                self.transform = dst_transform
                self.nodata = src.nodata
        except rasterio.errors.RasterioError as e:
            raise RuntimeError(
                f"{path} could not be fully read -- this usually means the file is "
                f"corrupted or truncated (e.g. an interrupted download saved as if "
                f"it completed). Underlying error: {e}. Re-download the DEM (see "
                "get_dem.py) and confirm the download actually finishes before "
                "retrying."
            ) from e
        finally:
            src.close()

        # Inverse affine transform: UTM (x,y) -> fractional (col,row)
        self.inv_transform = ~self.transform
        self.height, self.width = self.array.shape

    def elevations(self, xs, ys):
        """
        Vectorised bilinear elevation lookup.
        xs, ys : 1D numpy arrays of UTM coordinates (same length)
        returns : 1D numpy array of elevations (metres)
        """
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)

        cols, rows = self.inv_transform * (xs, ys)

        # Bounds check BEFORE clipping — catches points that fall outside
        # the DEM's raster footprint entirely (e.g. an inland-offset radar
        # placement that lands past the tile's edge). Without this, clip()
        # would silently reuse the edge pixel's elevation and the bad z_i
        # would quietly corrupt every LOS check that radar participates in.
        # This is a different failure mode than the NODATA check below —
        # NODATA catches "inside the raster but marked as void", this
        # catches "outside the raster's footprint" — both need separate
        # checks because clip()/mode="nearest" would otherwise mask both.
        out_of_bounds = (
            (cols < 0) | (cols > self.width - 1) |
            (rows < 0) | (rows > self.height - 1)
        )
        if np.any(out_of_bounds):
            n_bad = int(np.sum(out_of_bounds))
            bad_idx = np.where(out_of_bounds)[0][:5]
            bad_coords = [(float(xs[i]), float(ys[i])) for i in bad_idx]
            raise RuntimeError(
                f"{n_bad} sample(s) fall outside DEM coverage entirely "
                f"(not just NODATA — outside the raster footprint). "
                f"First offending coords (UTM): {bad_coords}. "
                f"{DEM_FILE} does not fully cover the AOI + inland offset "
                "+ largest radar range. Get a DEM tile with more margin."
            )

        cols = np.clip(cols, 0, self.width - 1.001)
        rows = np.clip(rows, 0, self.height - 1.001)

        # map_coordinates expects [row_coords, col_coords], order=1 = bilinear
        z = map_coordinates(self.array, [rows, cols], order=1, mode="nearest")

        if self.nodata is not None:
            bad = np.isclose(z, self.nodata)
            if np.any(bad):
                raise RuntimeError(
                    f"{np.sum(bad)} sample(s) hit NODATA in the DEM. "
                    "Border/radar geometry likely extends outside DEM coverage — "
                    "check that punjab_dem.tif fully covers the AOI plus a margin "
                    "at least as large as the largest radar range."
                )
        return z

    def elevation_at(self, x, y):
        return float(self.elevations(np.array([x]), np.array([y]))[0])


# ══════════════════════════════════════════════════════════════
#  3D GEOMETRY  —  curvature, LOS, FOV
#  Direct implementation of blocks 1-4 in the math reference doc.
# ══════════════════════════════════════════════════════════════

def h_bulge(s, d_horiz):
    """
    Block 1: Earth curvature bulge above the flat chord, at ray
    parameter s in [0,1]. Zero at both ends, peaks at the midpoint.
    h_bulge = s(1-s) * d_horiz^2 / (2 * R_eff)
    """
    return s * (1.0 - s) * d_horiz ** 2 / (2.0 * R_EFF)


def los_clear(rx, ry, rz, px, py, pz, dem, d_horiz):
    """
    Block 2: LOS ray test, curvature-corrected.

    Marches along the straight-line ray at one sample per DEM pixel and
    checks the ray never dips below (terrain + curvature bulge).

    Edge case handled: d_horiz == 0 (radar directly above/below the
    point — e.g. a radar and boundary point that coincide in x,y). The
    original formula gives S = ceil(0/30) = 0 ray samples, which would
    silently pass with no check at all. Here we do a direct single
    comparison instead: LOS is only blocked if the point sits below the
    radar's own footprint elevation, which can't happen since z_i and
    z_j both derive from the same terrain at the same (x,y).
    """
    if d_horiz < 1e-6:
        return True

    # Sample at twice the DEM's pixel resolution, not once per pixel.
    # Sampling at exactly DEM_PIXEL_M spacing on a DEM of that same
    # resolution risks aliasing — a narrow ridge peak can fall between
    # two consecutive samples and get missed ("false LOS clear"). The
    # max(4, ...) floor also fixes a short-range degenerate case: at
    # d_horiz just under DEM_PIXEL_M, max(1, ...) would give S=1, i.e.
    # only the target point itself is checked, with zero samples along
    # the way to catch a nearby obstruction.
    S = max(4, math.ceil(d_horiz / (DEM_PIXEL_M / 2.0)))
    s = np.arange(1, S + 1, dtype=np.float64) / S     # s in (0, 1], step 1/S

    ray_z = rz + s * (pz - rz)
    xs = rx + s * (px - rx)
    ys = ry + s * (py - ry)
    terrain_z = dem.elevations(xs, ys)

    if ENABLE_EARTH_CURVATURE:
        bulge = h_bulge(s, d_horiz)
    else:
        bulge = 0.0

    # LOS_EPS_M tolerance: at s=1, ray_z(1) == pz and terrain_z(1) should
    # equal pz exactly (the point sits on the terrain by construction),
    # so this comparison is a true equality in theory. Without slack,
    # DEM bilinear interpolation noise or float rounding can flip a
    # genuinely-clear ray to "blocked" right at the target. The eps only
    # matters at that boundary; it's too small to hide a real obstruction.
    return bool(np.all(ray_z + LOS_EPS_M >= terrain_z + bulge))


def in_fov(rz, pz, d_horiz, theta_min, theta_max):
    """
    Block 3: FOV / elevation angle check.
    theta = arctan(Δz / d_horiz), degrees. Negative = looking downward.

    Edge case handled: d_horiz == 0 -> arctan(Δz/0) is undefined in the
    doc's formula. We resolve it as straight up/down (+-90 deg) rather
    than raising, since a radar and point at the same (x,y) is a valid
    (if degenerate) geometric configuration.
    """
    if d_horiz < 1e-6:
        theta_deg = 90.0 if pz >= rz else -90.0
    else:
        theta_deg = math.degrees(math.atan2(pz - rz, d_horiz))
    return theta_min <= theta_deg <= theta_max


def in_azimuth(dx, dy, boresight_deg, halfwidth_deg):
    """
    Horizontal FOV check — the piece the original in_fov() never had.

    dx, dy        : horizontal vector from radar to point (px - rx, py - ry),
                    same UTM axes used everywhere else in this file.
    boresight_deg : direction the radar faces, atan2(dy,dx) convention
                    (0 deg = +x/East, 90 deg = +y/North, CCW-positive).
    halfwidth_deg : +/- sector half-width around boresight. E.g. 90 deg
                    gives a 180 deg-wide forward hemisphere; 180 deg
                    effectively disables the check (full circle).

    Edge case: dx == dy == 0 (point coincides with radar in x,y) has no
    defined bearing. Treat as "in azimuth" — same convention as in_fov()
    treating the equivalent d_horiz==0 case as a degenerate pass-through
    rather than a hard fail, since range/LOS already handle that geometry
    correctly on their own.
    """
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return True

    bearing_deg = math.degrees(math.atan2(dy, dx))

    # Wrap the difference into (-180, 180] before comparing, so a sector
    # straddling the +/-180 seam (e.g. boresight = 175 deg) doesn't
    # falsely fail for points just past the wraparound.
    diff = (bearing_deg - boresight_deg + 180.0) % 360.0 - 180.0
    return abs(diff) <= halfwidth_deg


def utm_epsg_for_lonlat(lon, lat):
    """
    Return the EPSG code of the UTM zone that contains (lon, lat).

    UTM divides the globe into 6°-wide longitude zones, each with its
    own "central meridian" where the projection is most accurate —
    distortion grows the further you get from it. Zone 43N's central
    meridian is 75°E, which is why 32643 worked for Punjab specifically;
    it is NOT a universal constant. This computes the right zone for
    whatever border geojson is actually loaded, so the same code is
    correct for any Indian state (India spans UTM zones ~42N-46N),
    or anywhere else in the world.

    zone formula: standard UTM zone numbering, 1-60, width 6° each,
    zone 1 starting at -180°. Northern hemisphere -> EPSG 326xx,
    southern hemisphere -> EPSG 327xx (not expected here, but correct
    regardless of which hemisphere BORDER_FILE turns out to be in).
    """
    zone = int((lon + 180) / 6) + 1
    zone = max(1, min(60, zone))   # clamp, in case of a coordinate right at +-180
    return (32600 if lat >= 0 else 32700) + zone


# ══════════════════════════════════════════════════════════════
#  STEP 1 — LOAD AND STITCH ALL OSM SEGMENTS  (unchanged from v1.2)
# ══════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("  BUILD_MATRIX v1.3  —  3D Terrain-Aware Coverage Model (external placement)")
print("═" * 60)

print(f"\n[1] Loading {BORDER_FILE} ...")
border_gdf = gpd.read_file(BORDER_FILE)
print(f"    Raw features in GeoJSON : {len(border_gdf)}")

# Auto-detect the correct UTM zone from the border's own centroid,
# using its native CRS (almost always WGS84 lon/lat for OSM exports)
# BEFORE reprojecting anything. This replaces the old hardcoded 32643.
border_wgs84 = border_gdf.to_crs(epsg=4326).geometry
merged_wgs84 = border_wgs84.union_all() if hasattr(border_wgs84, "union_all") else border_wgs84.unary_union
centroid_lonlat = merged_wgs84.centroid
TARGET_EPSG = utm_epsg_for_lonlat(centroid_lonlat.x, centroid_lonlat.y)
print(f"    Centroid (lon, lat)     : ({centroid_lonlat.x:.3f}, {centroid_lonlat.y:.3f})")
print(f"    Auto-detected UTM zone  : EPSG:{TARGET_EPSG}")

border_gdf = border_gdf.to_crs(epsg=TARGET_EPSG)
line = linemerge(unary_union(border_gdf.geometry))

if line.geom_type != "LineString":
    raise RuntimeError(
        f"linemerge produced {line.geom_type} — segments may have gaps. "
        "Inspect the GeoJSON for discontinuities."
    )

print(f"    Features merged         : {len(border_gdf)} → 1 LineString")
print(f"    Total border length     : {line.length / 1000:.1f} km")


# ══════════════════════════════════════════════════════════════
#  STEP 2 — DISCRETISE INTO BOUNDARY POINTS  (unchanged from v1.2)
# ══════════════════════════════════════════════════════════════

print(f"\n[2] Discretising at {INTERVAL_M} m intervals ...")
n_pts = int(line.length // INTERVAL_M)
bpts  = [line.interpolate(i * INTERVAL_M) for i in range(n_pts + 1)]
print(f"    K = {len(bpts)} boundary points")


# ══════════════════════════════════════════════════════════════
#  STEP 3 — LOAD RADAR PLACEMENT + TYPE CONFIG
#  Placement (x, y, range_m, type, boresight_deg, ground_z, h_ant, z) is no
#  longer computed here -- it was already fully computed once by
#  optimize_placement.py and is loaded verbatim. This is now an inter-file
#  contract between two independently-runnable scripts (not an in-memory
#  guarantee within one script), so the loaded list is validated fail-fast
#  before use, matching DEMReader.elevations()'s diagnostic style.
# ══════════════════════════════════════════════════════════════

REQUIRED_RADAR_FIELDS = (
    "name", "x", "y", "range_m", "type",
    "boresight_deg", "ground_z", "h_ant", "z",
)


def load_radar_placement(path):
    """
    Loads a JSON list of radar dicts (schema produced by
    optimize_placement.py's write_placement_json(), identical to what this
    script itself used to build in-memory as `radars`). Validates every
    radar has all required fields -- fail-fast, naming the offending radar
    (by its list index and name, if present) and the missing field(s),
    since a malformed radar_meta_optimized.json would otherwise surface as
    a confusing KeyError deep inside STEP 5.
    """
    with open(path) as f:
        radars = json.load(f)

    if not isinstance(radars, list):
        raise RuntimeError(
            f"{path} must contain a JSON list of radar dicts "
            f"(got {type(radars).__name__})."
        )
    if len(radars) == 0:
        raise RuntimeError(f"{path} contains zero radars -- nothing to build a matrix for.")

    for idx, r in enumerate(radars):
        missing = [field for field in REQUIRED_RADAR_FIELDS if field not in r]
        if missing:
            label = r.get("name", f"index {idx}")
            raise RuntimeError(
                f"Radar '{label}' in {path} is missing required field(s): {missing}. "
                f"Expected schema: {list(REQUIRED_RADAR_FIELDS)}. "
                "This file is produced by optimize_placement.py -- if it was "
                "hand-edited or came from an older/incompatible run, regenerate it."
            )

    return radars


def load_type_config(path):
    """
    Loads radar_type_config.json (schema produced by optimize_placement.py's
    write_type_config_json()): a dict keyed by type name, each value holding
    count/range_m/h_ant/theta_min/theta_max/az_halfwidth/energy_budget/
    cooling_L/cooling_C. Only theta_min/theta_max/az_halfwidth are consumed
    by this script (in STEP 5, gated on ENABLE_FOV_CHECK/ENABLE_AZIMUTH_CHECK)
    -- h_ant/range_m/count/energy_budget/cooling_L/cooling_C are irrelevant
    here (h_ant is already baked into each radar's loaded "z" field; the rest
    are scheduler-only config).
    """
    with open(path) as f:
        type_config = json.load(f)

    if not isinstance(type_config, dict):
        raise RuntimeError(
            f"{path} must contain a JSON object keyed by type name "
            f"(got {type(type_config).__name__})."
        )
    return type_config


print(f"\n[3] Loading radar placement from {INPUT_PLACEMENT} ...")
radars = load_radar_placement(INPUT_PLACEMENT)
print(f"    Loaded {len(radars)} radars (placement already computed by optimize_placement.py)")

print(f"    Loading per-type FOV/azimuth config from {INPUT_TYPE_CONFIG} ...")
type_config = load_type_config(INPUT_TYPE_CONFIG)
print(f"    Loaded config for {len(type_config)} type(s): {sorted(type_config.keys())}")


# ══════════════════════════════════════════════════════════════
#  STEP 4 — DEM-DERIVE Z FOR BOUNDARY POINTS
#  Radar elevations (ground_z, h_ant, z) are no longer computed here --
#  they arrive already-populated in the loaded radar list (STEP 3), computed
#  once by optimize_placement.py. Only boundary-point elevations still need
#  a DEM pass: z_j = DEM(x_j,y_j) + H_TARGET.
# ══════════════════════════════════════════════════════════════

print(f"\n[4] Loading DEM and assigning elevations ...")
dem = DEMReader(DEM_FILE, target_epsg=TARGET_EPSG)

bpt_xs = np.array([p.x for p in bpts])
bpt_ys = np.array([p.y for p in bpts])
bpt_ground_z = dem.elevations(bpt_xs, bpt_ys)
bpt_z = bpt_ground_z + H_TARGET   # z_j, per point

print(f"    Boundary elevations: {bpt_ground_z.min():.1f} – {bpt_ground_z.max():.1f} m (ground)")

radar_ground_z = np.array([r["ground_z"] for r in radars])
print(f"    Radar elevations   : {radar_ground_z.min():.1f} – {radar_ground_z.max():.1f} m "
      f"(ground, loaded from {INPUT_PLACEMENT})")

type_counts = {}
for r in radars:
    type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1
print("    " + "  |  ".join(f"{n} × {t}" for t, n in sorted(type_counts.items())))


# ══════════════════════════════════════════════════════════════
#  STEP 5 — BUILD 3D COVERAGE MATRIX  C[i][j]
#  Block 4: C_ij = range AND LOS AND FOV
#
#  Ordering for speed: range check first (cheap, numpy, rejects the
#  vast majority of pairs since radar range << border length), then
#  FOV (cheap, one atan2 per surviving pair), then LOS last (the
#  expensive ray-marched check), only run on genuine candidates.
# ══════════════════════════════════════════════════════════════

print(f"\n[5] Building {len(radars)} × {len(bpts)} 3D coverage matrix ...")
print(f"    Earth curvature : {'ON' if ENABLE_EARTH_CURVATURE else 'OFF'}")
print(f"    FOV (elevation) : {'ON' if ENABLE_FOV_CHECK else 'OFF (pending professor input)'}")
print(f"    FOV (azimuth)   : {'ON' if ENABLE_AZIMUTH_CHECK else 'OFF (pending professor input)'}")

C_matrix = []
total_coverage_cells = 0

for radar in radars:
    rx, ry, rz = radar["x"], radar["y"], radar["z"]
    rng = radar["range_m"]

    dx = bpt_xs - rx
    dy = bpt_ys - ry
    d_horiz_all = np.hypot(dx, dy)
    d_3d_all = np.sqrt(d_horiz_all ** 2 + (bpt_z - rz) ** 2)

    # Condition 1: 3D slant range <= radar range
    candidates = np.where(d_3d_all <= rng)[0]

    row = [0] * len(bpts)

    if ENABLE_FOV_CHECK or ENABLE_AZIMUTH_CHECK:
        rtype = radar["type"]
        if rtype not in type_config:
            raise RuntimeError(
                f"Radar '{radar['name']}' has type '{rtype}', which is not a key in "
                f"{INPUT_TYPE_CONFIG} (available types: {sorted(type_config.keys())}). "
                f"{INPUT_PLACEMENT} and {INPUT_TYPE_CONFIG} are inconsistent with each "
                "other -- they must come from the same optimize_placement.py run. "
                "Regenerate both together."
            )

    if ENABLE_FOV_CHECK:
        theta_min = type_config[rtype]["theta_min"]
        theta_max = type_config[rtype]["theta_max"]

    if ENABLE_AZIMUTH_CHECK:
        boresight_deg = radar["boresight_deg"]
        az_halfwidth = type_config[rtype]["az_halfwidth"]

    for j in candidates:
        d_h = d_horiz_all[j]

        # Condition 3a: elevation FOV (cheap — check before the ray march)
        if ENABLE_FOV_CHECK:
            if not in_fov(rz, bpt_z[j], d_h, theta_min, theta_max):
                continue

        # Condition 3b: azimuth FOV (cheap — check before the ray march).
        # Independent toggle from elevation: a radar can have a full
        # vertical window but still be a directional/sector antenna
        # horizontally, or vice versa.
        if ENABLE_AZIMUTH_CHECK:
            if not in_azimuth(dx[j], dy[j], boresight_deg, az_halfwidth):
                continue

        # Condition 2: LOS (expensive — ray march against DEM)
        if los_clear(rx, ry, rz, bpt_xs[j], bpt_ys[j], bpt_z[j], dem, d_h):
            row[j] = 1

    C_matrix.append(row)
    covered = sum(row)
    total_coverage_cells += covered
    pct = 100 * covered / len(bpts)
    print(f"    {radar['name']:12s}  range={rng/1000:2.0f} km  "
          f"candidates={len(candidates):4d}  covers {covered:4d}/{len(bpts)} pts  ({pct:5.1f}%)")

uncovered = sum(
    1 for j in range(len(bpts))
    if all(C_matrix[i][j] == 0 for i in range(len(radars)))
)
print(f"\n    Total coverage cells    : {total_coverage_cells}")
print(f"    Permanently uncoverable : {uncovered} / {len(bpts)} points "
      f"({100*uncovered/len(bpts):.1f}%)")
if uncovered > 0:
    print("    ⚠  Some points cannot be covered by ANY radar (range, LOS, or FOV).")
    print("       Scheduler will route into Phase 1 (minimise void).")
else:
    print("    ✓  Every point is reachable by at least one radar.")


# ══════════════════════════════════════════════════════════════
#  STEP 6 — EXPORT  (same contract as v1.2 — scheduler.py is untouched)
# ══════════════════════════════════════════════════════════════

print(f"\n[6] Exporting ...")
with open(OUTPUT_MATRIX, "w") as f:
    json.dump(C_matrix, f)
print(f"    Saved matrix   → {OUTPUT_MATRIX}  "
      f"({len(radars)} radars × {len(bpts)} points)")

with open(OUTPUT_META, "w") as f:
    json.dump(radars, f, indent=2)
print(f"    Saved metadata → {OUTPUT_META}  (pass-through re-export of {INPUT_PLACEMENT}, same schema)")

print(f"\n[Done]  Run scheduler_1.2.py next — c_matrix.json format is unchanged.\n")
