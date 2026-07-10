"""
optimize_placement.py
======================
V1 radar placement optimizer — Budget-Constrained Maximum Coverage Location
Problem (MCLP), solved exactly via OR-Tools CP-SAT, over a discretized
candidate-site lattice generated from the border geometry + DEM.

Why discretize instead of optimizing continuous (x,y): line-of-sight is a
non-smooth, black-box terrain oracle (a 1m shift across a ridge can flip
LOS for an entire arc of boundary points), so gradient/continuous methods
have no meaningful landscape to climb. Exact ILP over a finite candidate
set sidesteps that entirely and gives a provable optimum (or bound) instead
of a metaheuristic's "best found."

What this script does:
  1. Asks the user (interactively, or via --config) how many radar TYPES
     exist and, per type: name, exact count to place, max range, antenna
     height, elevation FOV window, azimuth half-width, plus Energy/Cooling
     (L, C) values that this script does NOT use itself but saves for a
     future updated scheduler_1_2.py.
  2. Loads the border geojson + DEM (same auto-UTM-detection /
     bilinear-elevation machinery as build_matrix_1.3.py), and builds a
     candidate lattice of (x, y, boresight_deg) sites by calling
     build_matrix_1.3.py's inland_point() over a grid of along-border
     positions x inland depth rings — instead of once per hardcoded radar.
  3. Evaluates coverage (range AND elevation-FOV AND azimuth-FOV AND LOS,
     the same 4 checks and check-order as build_matrix_1.3.py's STEP 5) for
     every (candidate, type) pair against every boundary point.
  4. Solves an exact-count MCLP with CP-SAT: choose exactly N_t sites for
     each type t, at most one radar per site, minimum inter-site
     separation, maximizing total covered boundary points.
  5. Writes radar_meta_optimized.json (same schema as build_matrix_1.3.py's
     radar_meta.json) and radar_type_config.json (Energy/Cooling per type,
     for later scheduler use).

What this script deliberately does NOT do (out of scope for this task):
  - It does NOT modify build_matrix_1.3.py, scheduler_1_2.py, or
    generate_czml.py.
  - It does NOT produce c_matrix.json — that stays build_matrix_1.3.py's
    job. Wiring build_matrix_1.3.py to consume radar_meta_optimized.json as
    an input (instead of generating its own RADAR_SPECS placement) is
    future work, a separate task.
  - Boresight and elevation are NEVER decision variables — both are fixed
    per candidate at generation time (boresight from inland_point()'s
    existing "face back at the border" formula; elevation from a DEM
    lookup), exactly mirroring how build_matrix_1.3.py already treats them.

Generalization: no Punjab-specific or fixed-type-name assumptions are
hardcoded here. BORDER_FILE/DEM_FILE below are the only dataset-specific
inputs (hand-edit them when switching datasets, exactly as
build_matrix_1.3.py already expects) — everything else (type names,
counts, hardware parameters) comes from the user.

The physics/geometry functions below (DEMReader, h_bulge, los_clear,
in_fov, in_azimuth, inland_point, utm_epsg_for_lonlat) and the border
loading/discretization logic are intentionally duplicated from
build_matrix_1.3.py rather than imported from it — build_matrix_1.3.py is
a top-level script whose Steps 1-6 execute at import time, and this task
is explicitly scoped to not modify it. This mirrors the same duplication
pattern already established between build_matrix_1.3.py and
generate_czml.py (see CLAUDE.md's "Cross-file invariants" section).

Usage:
    pip install numpy geopandas shapely rasterio scipy ortools
    python optimize_placement.py                  # interactive wizard
    python optimize_placement.py --config types.json --reference-point 74.8723,31.6340
                                                    # non-interactive, for repeatable runs
    (--reference-point is a lon,lat point known to lie on YOUR side of the
    border, e.g. a city in your own territory -- see inland_point()'s
    docstring for why this is needed. Prompted for interactively if omitted.)
"""

import argparse
import json
import math
import sys

import numpy as np
import geopandas as gpd
from shapely.ops import linemerge, unary_union, split
from shapely.geometry import Point, Polygon, LineString
from shapely.prepared import prep

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from ortools.sat.python import cp_model

# Console output below uses box-drawing/checkmark characters (matching
# build_matrix_1.3.py's and scheduler_1_2.py's existing print style) --
# force UTF-8 on stdout so this doesn't crash under a plain Windows
# console (cp1252), which can't encode them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION — dataset files (hand-edit when switching datasets)
# ══════════════════════════════════════════════════════════════

BORDER_FILE = "punjab_border.geojson"
DEM_FILE    = "punjab_dem.tif"

OUTPUT_PLACEMENT   = "radar_meta_optimized.json"
OUTPUT_TYPE_CONFIG = "radar_type_config.json"

# MUST match build_matrix_1.3.py's INTERVAL_M, or the boundary points used
# for coverage here won't line up with what the rest of the pipeline
# eventually builds c_matrix.json against (same cross-file invariant
# documented in CLAUDE.md).
INTERVAL_M = 250

# ── Candidate lattice (site generation) ─────────────────────────
# Along-border spacing between candidate positions. Denser than
# INTERVAL_M is unnecessary -- candidates this close together produce
# near-identical coverage sets, and MIN_SEPARATION_M below excludes
# co-locating two radars this close anyway.
CANDIDATE_SPACING_M = 500

# Inland depth "rings" a candidate can be offset at, in metres. Brackets
# build_matrix_1.3.py's single fixed INLAND_M=2000 value with a spread of
# plausible siting depths.
CANDIDATE_DEPTH_RINGS_M = [1000, 2000, 4000, 8000]

# Minimum allowed 2D distance between any two INSTALLED radars, enforced
# as a CP-SAT mutual-exclusion constraint (not a candidate filter) --
# prevents the solver placing two radars at effectively the same spot.
MIN_SEPARATION_M = 750

# Padding around the border line's own bounding box used to build the
# "our side" territory polygon (see build_territory_polygon()) -- must be
# large enough that no candidate at any CANDIDATE_DEPTH_RINGS_M depth can
# fall outside it. Same margin philosophy get_dem.py already uses for DEM
# coverage (max candidate depth + generous slack).
TERRITORY_MARGIN_M = max(CANDIDATE_DEPTH_RINGS_M) + 5_000

# Coarse grid-bucket pre-thin before the expensive coverage evaluation.
# Performance optimization only -- correctness comes from the CP-SAT
# model's own exact pairwise MIN_SEPARATION_M constraint regardless of
# this flag.
DEDUPE_CANDIDATES = True

# CP-SAT wall-clock budget for the placement solve, same role/units as
# scheduler_1_2 (1).py's TIME_LIMIT.
TIME_LIMIT = 300.0

# ── Earth curvature (copied from build_matrix_1.3.py) ───────────
ENABLE_EARTH_CURVATURE = True
R_E   = 6_371_000.0          # mean Earth radius, m
K_REFRACTION = 4.0 / 3.0     # standard radio refraction factor
R_EFF = K_REFRACTION * R_E   # ~ 8,494,667 m

# ── LOS ray march (copied from build_matrix_1.3.py) ─────────────
DEM_PIXEL_M = 30.0
LOS_EPS_M   = 0.5

# ── FOV toggles (copied from build_matrix_1.3.py; keep matched to
#    whatever build_matrix_1.3.py itself uses, same cross-file invariant
#    already documented in CLAUDE.md) ────────────────────────────
ENABLE_FOV_CHECK     = True
ENABLE_AZIMUTH_CHECK = True

# Target height above ground for boundary points (copied from
# build_matrix_1.3.py's H_TARGET).
H_TARGET = 0.0


# ══════════════════════════════════════════════════════════════
#  HELPERS copied verbatim from scheduler_1_2 (1).py for a consistent
#  look/feel on the solve portion of this script's output.
# ══════════════════════════════════════════════════════════════

def banner(title, subtitle=""):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(f"{'─'*60}")


def status_label(code):
    return {
        cp_model.OPTIMAL:    "OPTIMAL",
        cp_model.FEASIBLE:   "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN:    "UNKNOWN (time-out?)",
    }.get(code, "UNKNOWN")


# ══════════════════════════════════════════════════════════════
#  CRS AUTO-DETECTION + BORDER LOADING
#  Verbatim port of build_matrix_1.3.py's utm_epsg_for_lonlat() and its
#  STEP 1 / STEP 2 (inline in the original; wrapped into functions here).
# ══════════════════════════════════════════════════════════════

def utm_epsg_for_lonlat(lon, lat):
    """
    Return the EPSG code of the UTM zone that contains (lon, lat).
    Standard UTM zone numbering, 1-60, width 6 deg each, zone 1 starting
    at -180 deg. Northern hemisphere -> EPSG 326xx, southern -> 327xx.
    """
    zone = int((lon + 180) / 6) + 1
    zone = max(1, min(60, zone))
    return (32600 if lat >= 0 else 32700) + zone


def load_border_line(border_file):
    """
    Verbatim port of build_matrix_1.3.py STEP 1: load the border geojson,
    auto-detect its UTM zone from its own centroid, reproject, and merge
    all segments into one continuous LineString.
    Returns (line, target_epsg).
    """
    print(f"[1] Loading {border_file} ...")
    border_gdf = gpd.read_file(border_file)
    print(f"    Raw features in GeoJSON : {len(border_gdf)}")

    border_wgs84 = border_gdf.to_crs(epsg=4326).geometry
    merged_wgs84 = (
        border_wgs84.union_all() if hasattr(border_wgs84, "union_all")
        else border_wgs84.unary_union
    )
    centroid_lonlat = merged_wgs84.centroid
    target_epsg = utm_epsg_for_lonlat(centroid_lonlat.x, centroid_lonlat.y)
    print(f"    Centroid (lon, lat)     : ({centroid_lonlat.x:.3f}, {centroid_lonlat.y:.3f})")
    print(f"    Auto-detected UTM zone  : EPSG:{target_epsg}")

    border_gdf = border_gdf.to_crs(epsg=target_epsg)
    line = linemerge(unary_union(border_gdf.geometry))

    if line.geom_type != "LineString":
        raise RuntimeError(
            f"linemerge produced {line.geom_type} — segments may have gaps. "
            "Inspect the GeoJSON for discontinuities."
        )

    print(f"    Features merged         : {len(border_gdf)} -> 1 LineString")
    print(f"    Total border length     : {line.length/1000:.1f} km")
    return line, target_epsg


def discretize_boundary(line, interval_m):
    """
    Verbatim port of build_matrix_1.3.py STEP 2: interpolate boundary
    points every interval_m metres along the merged border line.
    Returns a list of shapely Points.
    """
    print(f"[2] Discretising at {interval_m} m intervals ...")
    n_pts = int(line.length // interval_m)
    bpts = [line.interpolate(i * interval_m) for i in range(n_pts + 1)]
    print(f"    K = {len(bpts)} boundary points")
    return bpts


# ══════════════════════════════════════════════════════════════
#  DEM READER — verbatim port of build_matrix_1.3.py's DEMReader
# ══════════════════════════════════════════════════════════════

class DEMReader:
    def __init__(self, path, target_epsg):
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
                self.array = src.read(1).astype(np.float64)
                self.transform = src.transform
                self.nodata = src.nodata
            else:
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

        self._path = path
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
                f"{self._path} does not fully cover the AOI + candidate "
                "lattice + largest radar range. Get a DEM tile with more margin, "
                "or shrink CANDIDATE_DEPTH_RINGS_M."
            )

        cols = np.clip(cols, 0, self.width - 1.001)
        rows = np.clip(rows, 0, self.height - 1.001)

        z = map_coordinates(self.array, [rows, cols], order=1, mode="nearest")

        if self.nodata is not None:
            bad = np.isclose(z, self.nodata)
            if np.any(bad):
                raise RuntimeError(
                    f"{np.sum(bad)} sample(s) hit NODATA in the DEM. "
                    f"Border/candidate geometry likely extends outside DEM coverage — "
                    f"check that {self._path} fully covers the AOI plus a margin at "
                    "least as large as the largest radar range."
                )
        return z

    def elevation_at(self, x, y):
        return float(self.elevations(np.array([x]), np.array([y]))[0])


# ══════════════════════════════════════════════════════════════
#  3D GEOMETRY — curvature, LOS, FOV
#  Verbatim port of build_matrix_1.3.py's blocks 1-3 + inland_point().
# ══════════════════════════════════════════════════════════════

def h_bulge(s, d_horiz):
    """
    Earth curvature bulge above the flat chord, at ray parameter s in
    [0,1]. Zero at both ends, peaks at the midpoint.
    """
    return s * (1.0 - s) * d_horiz ** 2 / (2.0 * R_EFF)


def los_clear(rx, ry, rz, px, py, pz, dem, d_horiz):
    """
    LOS ray test, curvature-corrected. Marches along the straight-line ray
    at one sample per half-DEM-pixel and checks the ray never dips below
    (terrain + curvature bulge). See build_matrix_1.3.py for the full
    derivation/edge-case notes -- this is an exact port.
    """
    if d_horiz < 1e-6:
        return True

    S = max(4, math.ceil(d_horiz / (DEM_PIXEL_M / 2.0)))
    s = np.arange(1, S + 1, dtype=np.float64) / S

    ray_z = rz + s * (pz - rz)
    xs = rx + s * (px - rx)
    ys = ry + s * (py - ry)
    terrain_z = dem.elevations(xs, ys)

    if ENABLE_EARTH_CURVATURE:
        bulge = h_bulge(s, d_horiz)
    else:
        bulge = 0.0

    return bool(np.all(ray_z + LOS_EPS_M >= terrain_z + bulge))


def in_fov(rz, pz, d_horiz, theta_min, theta_max):
    """
    Elevation FOV check. theta = arctan(dz / d_horiz), degrees.
    d_horiz==0 resolves to straight up/down (+-90 deg) rather than raising.
    """
    if d_horiz < 1e-6:
        theta_deg = 90.0 if pz >= rz else -90.0
    else:
        theta_deg = math.degrees(math.atan2(pz - rz, d_horiz))
    return theta_min <= theta_deg <= theta_max


def in_azimuth(dx, dy, boresight_deg, halfwidth_deg):
    """
    Horizontal FOV check. dx,dy = horizontal vector from radar to point.
    boresight_deg/halfwidth_deg use the atan2(dy,dx) convention (0=+x/East,
    90=+y/North, CCW-positive). dx==dy==0 is treated as "in azimuth"
    (degenerate pass-through, matching in_fov()'s d_horiz==0 handling).
    """
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return True

    bearing_deg = math.degrees(math.atan2(dy, dx))
    diff = (bearing_deg - boresight_deg + 180.0) % 360.0 - 180.0
    return abs(diff) <= halfwidth_deg


def build_territory_polygon(line, ref_xy, margin_m):
    """
    Builds a polygon representing "our" side of the border (the side
    ref_xy is on), within margin_m of the border's own bounding box.

    A LineString alone can't answer "which side is X on" globally: the
    border is an OPEN curve whose two endpoints sit deep inside the AOI
    (not on its bounding-box edge), so near/beyond either endpoint you can
    walk from one side to the other WITHOUT crossing the line -- any test
    that only looks at the nearest point on the line (as earlier attempts
    at this did) is unreliable there. Closing the curve into a polygon and
    using point-in-polygon containment sidesteps this entirely, and still
    makes no assumption about the specific border/region.

    Steps: pad a box around the line's bounds; extend each endpoint
    outward along its own local tangent until it exits the box (via exact
    ray/boundary intersection, not a guessed distance); split the box with
    the resulting extended line (always 2 pieces for a simple curve); keep
    whichever piece contains ref_xy.
    """
    minx, miny, maxx, maxy = line.bounds
    box = Polygon([
        (minx - margin_m, miny - margin_m), (maxx + margin_m, miny - margin_m),
        (maxx + margin_m, maxy + margin_m), (minx - margin_m, maxy + margin_m),
    ])

    def extend_to_box_edge(p_end, p_near):
        dx, dy = p_end.x - p_near.x, p_end.y - p_near.y
        d_len = math.hypot(dx, dy)
        if d_len == 0:
            raise RuntimeError(
                "Border line has a zero-length segment at an endpoint -- "
                "cannot determine outward tangent direction to close the "
                "territory polygon."
            )
        dx /= d_len
        dy /= d_len
        far = Point(p_end.x + dx * 1_000_000.0, p_end.y + dy * 1_000_000.0)
        ray = LineString([p_end, far])
        inter = ray.intersection(box.boundary)
        if inter.is_empty:
            raise RuntimeError(
                "Could not extend the border line's endpoint out to the "
                "padded bounding box -- the box margin may be too small."
            )
        pts = list(inter.geoms) if hasattr(inter, "geoms") else [inter]
        pts.sort(key=lambda p: p_end.distance(p))
        return pts[0]

    coords = list(line.coords)
    ext_start = extend_to_box_edge(Point(coords[0]), Point(coords[1]))
    ext_end = extend_to_box_edge(Point(coords[-1]), Point(coords[-2]))
    extended_line = LineString([ext_start] + coords + [ext_end])

    pieces = list(split(box, extended_line).geoms)
    if len(pieces) != 2:
        raise RuntimeError(
            f"Splitting the padded bounding box by the (extended) border line "
            f"produced {len(pieces)} piece(s), expected exactly 2. The border "
            "geometry likely self-intersects or otherwise violates the "
            "simple-open-curve assumption this relies on -- inspect "
            f"{BORDER_FILE} for anomalies."
        )

    ref_pt = Point(*ref_xy)
    our_piece = next((p for p in pieces if p.contains(ref_pt)), None)
    if our_piece is None:
        raise RuntimeError(
            "--reference-point does not fall inside either side of the "
            "border within the padded AOI -- check the lon,lat value, or "
            "that it isn't essentially ON the border line itself."
        )
    return prep(our_piece), our_piece.area


def inland_point(line, fraction, offset_m, territory_polygon):
    """
    Returns (radar_point, boresight_deg), or (None, None) if neither
    offset direction lands inside territory_polygon (can happen very
    close to the border's endpoints, or where it curves tightly relative
    to offset_m -- safer to drop the candidate than guess). boresight_deg
    is the direction the site naturally "faces": the reverse of the
    outward offset normal, i.e. pointing from the inland-offset position
    back at the border segment it was placed to watch.
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
        return None, None

    tx /= t_len
    ty /= t_len
    nx, ny = ty, -tx

    cand_a = Point(pt.x + nx * offset_m, pt.y + ny * offset_m)
    cand_b = Point(pt.x - nx * offset_m, pt.y - ny * offset_m)

    if territory_polygon.contains(cand_a):
        radar_pt, fnx, fny = cand_a, nx, ny
    elif territory_polygon.contains(cand_b):
        radar_pt, fnx, fny = cand_b, -nx, -ny
    else:
        return None, None

    boresight_deg = math.degrees(math.atan2(-fny, -fnx))
    return radar_pt, boresight_deg


# ══════════════════════════════════════════════════════════════
#  CANDIDATE GENERATION
# ══════════════════════════════════════════════════════════════

def generate_candidates(line, spacing_m, depth_rings_m, territory_polygon):
    """
    Along-border x depth-ring candidate lattice. Calls inland_point() for
    every (pos_pct, offset_m) combination -- boresight is fixed here, at
    generation time, never a decision variable downstream.
    territory_polygon is the authoritative "our side" region (see
    build_territory_polygon()); candidates inland_point() can't place
    inside it from either offset direction are dropped.

    Returns (candidates, n_dropped) -- candidates is a list of dicts:
    {x, y, boresight_deg}.
    """
    n_steps = max(1, int(line.length // spacing_m))
    candidates = []
    n_dropped = 0
    for i in range(n_steps + 1):
        pos_pct = i / n_steps
        for offset_m in depth_rings_m:
            pt, boresight_deg = inland_point(line, pos_pct, offset_m, territory_polygon)
            if pt is None:
                n_dropped += 1
                continue
            candidates.append({"x": pt.x, "y": pt.y, "boresight_deg": boresight_deg})
    return candidates, n_dropped


def dedupe_candidates(candidates, min_sep_m):
    """
    Coarse grid-bucket thin: round (x,y) to a min_sep_m-sized cell, keep
    only the first candidate seen per cell. Performance optimization
    only -- correctness comes from the CP-SAT model's own exact pairwise
    separation constraint (pairwise_too_close()), which runs regardless.
    """
    seen = set()
    kept = []
    for c in candidates:
        cell = (round(c["x"] / min_sep_m), round(c["y"] / min_sep_m))
        if cell in seen:
            continue
        seen.add(cell)
        kept.append(c)
    return kept


def build_candidate_coverage(candidates, radar_types, bpt_xs, bpt_ys, bpt_z, dem):
    """
    For every candidate site x every user-defined type, evaluates coverage
    against every boundary point using the same range -> in_fov ->
    in_azimuth -> los_clear check order (cheap filters before the
    expensive ray-march) as build_matrix_1.3.py's STEP 5.

    Boundary-point elevations (bpt_z) are precomputed once by the caller
    and passed in -- never recomputed here. Candidate ground_z is computed
    once per candidate (independent of type, since (x,y) doesn't change);
    z = ground_z + h_ant is computed per (candidate, type) pair since
    h_ant varies by type even for the same (x,y).

    Returns:
      a_cov          : dict (candidate_index, type_name) -> set of
                        covered boundary-point indices (sparse)
      cand_ground_z  : 1D numpy array, one ground elevation per candidate
      uncoverable    : count of boundary points covered by NO
                        (candidate, type) combination at all
      useful_count   : dict type_name -> number of candidates that cover
                        >=1 point when equipped with that type
    """
    print(f"\n[5] Evaluating candidate coverage (range -> FOV -> azimuth -> LOS) ...")
    print(f"    Earth curvature : {'ON' if ENABLE_EARTH_CURVATURE else 'OFF'}")
    print(f"    FOV (elevation) : {'ON' if ENABLE_FOV_CHECK else 'OFF'}")
    print(f"    FOV (azimuth)   : {'ON' if ENABLE_AZIMUTH_CHECK else 'OFF'}")

    cand_xs = np.array([c["x"] for c in candidates])
    cand_ys = np.array([c["y"] for c in candidates])
    cand_ground_z = dem.elevations(cand_xs, cand_ys)

    K = len(bpt_xs)
    n_candidates = len(candidates)
    a_cov = {}
    covered_anywhere = np.zeros(K, dtype=bool)
    useful_count = {rt["name"]: 0 for rt in radar_types}

    progress_every = max(1, n_candidates // 10)

    for i, cand in enumerate(candidates):
        rx, ry = cand["x"], cand["y"]
        boresight_deg = cand["boresight_deg"]
        ground_z_i = cand_ground_z[i]

        dx = bpt_xs - rx
        dy = bpt_ys - ry
        d_horiz_all = np.hypot(dx, dy)

        for rtype in radar_types:
            name = rtype["name"]
            rz = ground_z_i + rtype["h_ant"]
            d_3d_all = np.sqrt(d_horiz_all ** 2 + (bpt_z - rz) ** 2)
            cand_j = np.where(d_3d_all <= rtype["range_m"])[0]

            covered = set()
            for j in cand_j:
                d_h = d_horiz_all[j]

                if ENABLE_FOV_CHECK:
                    if not in_fov(rz, bpt_z[j], d_h, rtype["theta_min"], rtype["theta_max"]):
                        continue

                if ENABLE_AZIMUTH_CHECK:
                    if not in_azimuth(dx[j], dy[j], boresight_deg, rtype["az_halfwidth"]):
                        continue

                if los_clear(rx, ry, rz, bpt_xs[j], bpt_ys[j], bpt_z[j], dem, d_h):
                    j_int = int(j)
                    covered.add(j_int)
                    covered_anywhere[j_int] = True

            a_cov[(i, name)] = covered
            if covered:
                useful_count[name] += 1

        if (i + 1) % progress_every == 0 or i == n_candidates - 1:
            print(f"    ... evaluated {i+1}/{n_candidates} candidates")

    for rtype in radar_types:
        name = rtype["name"]
        pct = 100 * useful_count[name] / n_candidates if n_candidates else 0.0
        print(f"    Type '{name}' (range={rtype['range_m']:.0f}m, "
              f"theta=[{rtype['theta_min']:.1f},{rtype['theta_max']:.1f}], "
              f"az=+/-{rtype['az_halfwidth']:.1f}): "
              f"{useful_count[name]}/{n_candidates} candidates cover >=1 pt ({pct:.1f}%)")

    uncoverable = int(K - covered_anywhere.sum())
    return a_cov, cand_ground_z, uncoverable, useful_count


def pairwise_too_close(candidates, min_sep_m):
    """
    Returns a list of (i, i2) candidate-index pairs (i < i2) whose 2D
    Euclidean distance is < min_sep_m, via a KDTree for scalability beyond
    a naive O(n^2) all-pairs check at a few thousand candidates.
    """
    coords = np.array([[c["x"], c["y"]] for c in candidates])
    tree = cKDTree(coords)
    return list(tree.query_pairs(r=min_sep_m))


# ══════════════════════════════════════════════════════════════
#  CP-SAT MODEL — Budget-Constrained Maximum Coverage (MCLP)
# ══════════════════════════════════════════════════════════════

def solve_placement(a_cov, num_candidates, type_names, type_counts, too_close_pairs, K, time_limit):
    """
    Exact-count MCLP:
      maximize   sum_j z_j
      subject to z_j <= sum_i sum_t a[i][t][j] * y[i][t]           for all j
                 sum_i y[i][t] == type_counts[t]                    for all t  (EXACT count)
                 sum_t y[i][t] <= 1                                 for all i  (one radar per site)
                 y[i][t] + y[i2][t'] <= 1 for close (i,i2), summed over t     (min separation)
    No boresight or elevation variables appear here -- both are already
    baked into a_cov by candidate-generation time.
    """
    banner("PLACEMENT SOLVE — Budget-Constrained Maximum Coverage (MCLP)",
           f"{num_candidates} candidate sites x {len(type_names)} types, K={K} points")

    model = cp_model.CpModel()
    I = range(num_candidates)
    T = range(len(type_names))

    y = [[model.NewBoolVar(f"y_{i}_{t}") for t in T] for i in I]
    z = [model.NewBoolVar(f"z_{j}") for j in range(K)]

    coverers_of_j = [[] for _ in range(K)]
    for i in I:
        for t_idx, tname in enumerate(type_names):
            for j in a_cov[(i, tname)]:
                coverers_of_j[j].append(y[i][t_idx])

    for j in range(K):
        if coverers_of_j[j]:
            model.Add(z[j] <= sum(coverers_of_j[j]))
        else:
            model.Add(z[j] == 0)

    for t_idx, tname in enumerate(type_names):
        model.Add(sum(y[i][t_idx] for i in I) == type_counts[tname])

    for i in I:
        model.Add(sum(y[i][t] for t in T) <= 1)

    # Equivalent to enforcing y[i][t]+y[i2][t']<=1 for every (t,t') pair,
    # given the per-site <=1 constraint above already makes "is site i
    # occupied at all" a single 0/1 quantity -- one constraint per
    # too-close pair instead of |T|^2 per pair.
    for (i, i2) in too_close_pairs:
        model.Add(sum(y[i][t] for t in T) + sum(y[i2][t] for t in T) <= 1)

    model.Maximize(sum(z))

    print(f"  Requested counts : " + ", ".join(f"{n}={type_counts[n]}" for n in type_names))
    print(f"  Time limit       : {time_limit}s")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)
    label = status_label(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"  Solver status: {label}")
        return {"status": label}

    covered = sum(int(solver.Value(z[j])) for j in range(K))
    void = K - covered
    print(f"\n  Solver status   : {label}")
    print(f"  Covered points  : {covered} / {K}  ({100*covered/K:.1f}%)")
    print(f"  Void points     : {void} / {K}  ({100*void/K:.1f}%)")

    placements = []
    for i in I:
        for t_idx, tname in enumerate(type_names):
            if solver.Value(y[i][t_idx]):
                placements.append({"candidate_index": i, "type": tname})

    return {"status": label, "covered": covered, "void": void, "placements": placements}


# ══════════════════════════════════════════════════════════════
#  RESULT ASSEMBLY
# ══════════════════════════════════════════════════════════════

def build_final_radars(placements, candidates, cand_ground_z, type_by_name):
    """
    Combines solved (candidate_index, type) placements with candidate
    geometry and per-type hardware config into final radar dicts, in the
    same order as `placements` (candidate-index / generation order).
    """
    radars = []
    for p in placements:
        i = p["candidate_index"]
        tname = p["type"]
        cand = candidates[i]
        rtype = type_by_name[tname]
        ground_z = float(cand_ground_z[i])
        h_ant = rtype["h_ant"]
        radars.append({
            "x": cand["x"],
            "y": cand["y"],
            "range_m": rtype["range_m"],
            "type": tname,
            "boresight_deg": cand["boresight_deg"],
            "ground_z": ground_z,
            "h_ant": h_ant,
            "z": ground_z + h_ant,
        })
    return radars


def assign_names(radars):
    """
    R{:02d}_{type} sequential across ALL placed radars, 1-indexed, in the
    order already established (candidate-index / generation order) --
    matches radar_meta.json's existing convention exactly (sequential
    across the whole list, not grouped by type: R01_Small, R02_Medium,
    R03_Large, ...).
    """
    named = []
    for idx, r in enumerate(radars, start=1):
        entry = {"name": f"R{idx:02d}_{r['type']}"}
        entry.update(r)
        named.append(entry)
    return named


# ══════════════════════════════════════════════════════════════
#  INTERACTIVE INPUT + --config LOADING
# ══════════════════════════════════════════════════════════════

def _prompt_float(prompt_text):
    while True:
        raw = input(prompt_text)
        try:
            return float(raw)
        except ValueError:
            print("    Please enter a number.")


def _prompt_int(prompt_text):
    while True:
        raw = input(prompt_text)
        try:
            return int(raw)
        except ValueError:
            print("    Please enter a whole number.")


def prompt_radar_types():
    """
    Interactive wizard collecting per-type placement AND scheduler-config
    fields. Energy/cooling_L/cooling_C are NOT used anywhere in this
    script's own optimization -- they're captured here and passed straight
    through to write_type_config_json() for a future updated
    scheduler_1_2.py to consume instead of its current hardcoded
    BUDGET_BY_TYPE / global L,C.
    """
    print("\n=== optimize_placement.py — Radar Type Configuration ===")
    n_types = _prompt_int("How many radar types? > ")

    types = []
    for idx in range(1, n_types + 1):
        print(f"\n--- Type {idx} of {n_types} ---")
        name = input('  Name (e.g. "Small"): > ').strip()
        count = _prompt_int("  Count (exact number of this type to place): > ")
        range_m = _prompt_float("  Max detection range (m): > ")
        h_ant = _prompt_float("  Antenna height above ground (m): > ")
        theta_min = _prompt_float("  Elevation angle window min (deg, e.g. -5.0): > ")
        theta_max = _prompt_float("  Elevation angle window max (deg, e.g. 20.0): > ")
        az_halfwidth = _prompt_float("  Azimuth beam half-width (deg, e.g. 60.0): > ")
        energy_budget = _prompt_float(
            "  Energy/budget value (max ON-slots per schedule window -- "
            "currently 24 in scheduler_1_2.py, but you may change that "
            "later; for scheduler config only, unused here): > "
        )
        cooling_L = _prompt_int("  Cooling L (max ON-slots per rolling window): > ")
        cooling_C = _prompt_int("  Cooling C (extra cool-down slots -> window = L+C): > ")

        types.append({
            "name": name, "count": count, "range_m": range_m, "h_ant": h_ant,
            "theta_min": theta_min, "theta_max": theta_max,
            "az_halfwidth": az_halfwidth, "energy_budget": energy_budget,
            "cooling_L": cooling_L, "cooling_C": cooling_C,
        })

    print("\nConfiguration summary:")
    for t in types:
        print(f"  {t['name']:8s}: count={t['count']}  range={t['range_m']:.0f}m  "
              f"h_ant={t['h_ant']:.1f}m  theta=[{t['theta_min']:.1f},{t['theta_max']:.1f}]  "
              f"az=+/-{t['az_halfwidth']:.1f}deg")
    proceed = input("Proceed? [Y/n] > ").strip().lower()
    if proceed not in ("", "y", "yes"):
        print("Aborted by user.")
        sys.exit(0)

    return types


def parse_reference_point(raw):
    """
    Parses "lon,lat" into a (lon, lat) float tuple. Used for both
    --reference-point CLI parsing and the interactive prompt, so both
    paths validate identically.
    """
    parts = raw.split(",")
    if len(parts) != 2:
        raise ValueError(f'Expected "lon,lat" (e.g. "74.8723,31.6340"), got "{raw}".')
    try:
        lon, lat = float(parts[0].strip()), float(parts[1].strip())
    except ValueError:
        raise ValueError(f'Expected two numbers "lon,lat", got "{raw}".')
    if not (-180.0 <= lon <= 180.0) or not (-90.0 <= lat <= 90.0):
        raise ValueError(
            f"lon must be in [-180,180] and lat in [-90,90], got lon={lon}, lat={lat}."
        )
    return lon, lat


def prompt_reference_point():
    """
    Asks for a single lon/lat point known to lie on "our" side of the
    border (e.g. a city inside the caller's own territory). Used to
    resolve which physical side of the border line inland_point() should
    offset candidates toward -- see inland_point()'s docstring for why
    this can't be inferred from the border geometry alone.
    """
    print("\n=== optimize_placement.py — Reference Point ===")
    print("  Enter a lon,lat point you know is on YOUR side of the border")
    print('  (e.g. a city in your own territory, like "74.8723,31.6340").')
    while True:
        raw = input("  Reference point (lon,lat): > ").strip()
        try:
            return parse_reference_point(raw)
        except ValueError as e:
            print(f"    {e}")


def load_config_file(path):
    """
    Loads a JSON list of per-type dicts, same schema the interactive
    wizard collects -- one dict per type with keys:
      name, count, range_m, h_ant, theta_min, theta_max, az_halfwidth,
      energy_budget, cooling_L, cooling_C
    Skips the wizard entirely; useful for repeatable runs since this repo
    has no test suite (per CLAUDE.md).
    """
    with open(path) as f:
        types = json.load(f)
    if not isinstance(types, list):
        raise RuntimeError(f"{path} must contain a JSON list of per-type objects.")
    return types


def validate_type_config(types):
    """Fail-fast checks, mirroring DEMReader.elevations()'s diagnostic style."""
    names = [t["name"] for t in types]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise RuntimeError(f"Duplicate radar type name(s): {dupes}. Names must be unique.")

    for t in types:
        name = t["name"]
        if t["count"] <= 0:
            raise RuntimeError(
                f"Type '{name}': count must be > 0 (got {t['count']}). "
                "Omit this type entirely instead of setting count=0."
            )
        if t["range_m"] <= 0:
            raise RuntimeError(f"Type '{name}': range_m must be > 0 (got {t['range_m']}).")
        if t["h_ant"] < 0:
            raise RuntimeError(f"Type '{name}': h_ant must be >= 0 (got {t['h_ant']}).")
        if t["theta_min"] >= t["theta_max"]:
            raise RuntimeError(
                f"Type '{name}': theta_min ({t['theta_min']}) must be < "
                f"theta_max ({t['theta_max']})."
            )
        if not (0 < t["az_halfwidth"] <= 180):
            raise RuntimeError(
                f"Type '{name}': az_halfwidth must be in (0, 180] (got {t['az_halfwidth']})."
            )
        if t["cooling_L"] > 24:
            print(
                f"    WARNING: type '{name}' cooling_L={t['cooling_L']} exceeds "
                "scheduler_1_2.py's current NUM_TIME_SLOTS=24 default -- fine if "
                "you've already changed that constant, otherwise this won't make "
                "sense to the scheduler yet."
            )


# ══════════════════════════════════════════════════════════════
#  OUTPUT WRITERS
# ══════════════════════════════════════════════════════════════

def write_placement_json(radars, path):
    """Same schema/format as build_matrix_1.3.py's radar_meta.json write."""
    with open(path, "w") as f:
        json.dump(radars, f, indent=2)


def write_type_config_json(types, path):
    """
    Keyed by type name, for a FUTURE updated scheduler_1_2.py to read
    instead of its current hardcoded BUDGET_BY_TYPE / global L,C. Storing
    cooling_L/cooling_C PER TYPE here is forward-looking -- today's
    scheduler applies cooling globally to every radar regardless of type;
    this file doesn't change that behavior on its own.
    """
    keyed = {
        t["name"]: {
            "count":         t["count"],
            "range_m":       t["range_m"],
            "h_ant":         t["h_ant"],
            "theta_min":     t["theta_min"],
            "theta_max":     t["theta_max"],
            "az_halfwidth":  t["az_halfwidth"],
            "energy_budget": t["energy_budget"],
            "cooling_L":     t["cooling_L"],
            "cooling_C":     t["cooling_C"],
        }
        for t in types
    }
    with open(path, "w") as f:
        json.dump(keyed, f, indent=2)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Generate an optimized exact-count radar placement via CP-SAT MCLP."
    )
    ap.add_argument(
        "--config", type=str, default=None,
        help="Path to a JSON file with radar type definitions, skipping the "
             "interactive wizard. See load_config_file()/prompt_radar_types() "
             "for the expected schema."
    )
    ap.add_argument(
        "--reference-point", type=str, default=None,
        help='"lon,lat" of a point known to lie on YOUR side of the border '
             "(e.g. a city in your own territory) -- resolves which physical "
             "side of the border line candidates are offset toward, since "
             "that can't be inferred from the border geometry alone. "
             "Skips the interactive prompt if passed."
    )
    args = ap.parse_args()

    print("\n" + "═" * 60)
    print("  OPTIMIZE_PLACEMENT v1  —  Budget-Constrained MCLP Radar Siting")
    print("═" * 60)

    if args.config:
        print(f"\n[0] Loading radar type config from {args.config} ...")
        type_configs = load_config_file(args.config)
    else:
        type_configs = prompt_radar_types()

    if args.reference_point:
        try:
            ref_lonlat = parse_reference_point(args.reference_point)
        except ValueError as e:
            raise RuntimeError(f"--reference-point: {e}")
    else:
        ref_lonlat = prompt_reference_point()

    validate_type_config(type_configs)
    type_names = [t["name"] for t in type_configs]
    type_counts = {t["name"]: t["count"] for t in type_configs}
    type_by_name = {t["name"]: t for t in type_configs}

    line, target_epsg = load_border_line(BORDER_FILE)
    bpts = discretize_boundary(line, INTERVAL_M)

    ref_gdf = gpd.GeoSeries([Point(*ref_lonlat)], crs="EPSG:4326").to_crs(epsg=target_epsg)
    ref_xy = (float(ref_gdf.iloc[0].x), float(ref_gdf.iloc[0].y))
    print(f"    Reference point (lon,lat): ({ref_lonlat[0]:.4f}, {ref_lonlat[1]:.4f})  "
          f"-> UTM ({ref_xy[0]:.1f}, {ref_xy[1]:.1f})")

    territory_polygon, territory_area_m2 = build_territory_polygon(line, ref_xy, TERRITORY_MARGIN_M)
    print(f"    Territory polygon (our side) : {territory_area_m2/1e6:.1f} km^2 "
          f"(margin {TERRITORY_MARGIN_M/1000:.0f} km)")

    print(f"\n[3] Generating candidate lattice ...")
    print(f"    Along-border spacing : {CANDIDATE_SPACING_M} m")
    print(f"    Inland depth rings   : {CANDIDATE_DEPTH_RINGS_M} m")
    candidates, n_dropped = generate_candidates(line, CANDIDATE_SPACING_M, CANDIDATE_DEPTH_RINGS_M, territory_polygon)
    print(f"    Raw candidates       : {len(candidates)}")
    if n_dropped:
        print(f"    Dropped (no valid offset direction): {n_dropped} -- neither offset "
              f"direction landed inside the territory polygon (near a border endpoint, "
              f"or too tight a bend relative to the offset depth).")

    if DEDUPE_CANDIDATES:
        candidates = dedupe_candidates(candidates, MIN_SEPARATION_M)
        print(f"    After dedup (<{MIN_SEPARATION_M}m) : {len(candidates)} survive")

    # Fatal: not even enough candidate SITES to satisfy the exact counts,
    # independent of coverage quality. Checked before the expensive DEM/
    # coverage pass or the solve.
    for tname in type_names:
        if type_counts[tname] > len(candidates):
            raise RuntimeError(
                f"Requested count for type '{tname}' ({type_counts[tname]}) exceeds "
                f"the total number of candidate sites generated ({len(candidates)}). "
                "Increase candidate density (lower CANDIDATE_SPACING_M or add more "
                "CANDIDATE_DEPTH_RINGS_M entries) or reduce the requested count."
            )

    print(f"\n[4] Loading DEM and assigning elevations ...")
    dem = DEMReader(DEM_FILE, target_epsg=target_epsg)

    bpt_xs = np.array([p.x for p in bpts])
    bpt_ys = np.array([p.y for p in bpts])
    bpt_ground_z = dem.elevations(bpt_xs, bpt_ys)
    bpt_z = bpt_ground_z + H_TARGET
    print(f"    Boundary elevations : {bpt_ground_z.min():.1f} – {bpt_ground_z.max():.1f} m (ground)")

    a_cov, cand_ground_z, uncoverable, useful_count = build_candidate_coverage(
        candidates, type_configs, bpt_xs, bpt_ys, bpt_z, dem
    )
    K = len(bpts)
    print(f"    Permanently uncoverable: {uncoverable} / {K} ({100*uncoverable/K:.1f}%)")

    if uncoverable == K:
        raise RuntimeError(
            "Every boundary point is uncoverable by every candidate/type combination -- "
            "check that the border/DEM files match and radar ranges are plausible."
        )

    # Soft warning: enough SITES exist (checked above), but maybe not
    # enough that actually cover anything for a given type -- the solver
    # can still legally satisfy the exact-count constraint with "dead
    # weight" placements that cover zero points.
    for tname in type_names:
        n_useful = useful_count[tname]
        if type_counts[tname] > n_useful:
            print(
                f"    WARNING: type '{tname}' requested {type_counts[tname]}, but only "
                f"{n_useful} candidate sites cover >=1 point when equipped with this "
                f"type. {type_counts[tname]-n_useful} placement(s) may cover zero points."
            )

    print(f"\n[6] Computing minimum-separation pairs (<{MIN_SEPARATION_M}m) ...")
    too_close_pairs = pairwise_too_close(candidates, MIN_SEPARATION_M)
    print(f"    {len(too_close_pairs)} candidate pairs excluded (mutual exclusion constraints)")

    result = solve_placement(
        a_cov, len(candidates), type_names, type_counts, too_close_pairs, K, TIME_LIMIT
    )

    if result["status"] not in ("OPTIMAL", "FEASIBLE"):
        raise RuntimeError(
            f"CP-SAT solve did not find a feasible placement (status: {result['status']}). "
            "This usually means MIN_SEPARATION_M is too large relative to candidate "
            "density and the requested exact counts, or the candidate lattice is too "
            "coarse. Try lowering MIN_SEPARATION_M, increasing candidate density, or "
            "reducing counts."
        )

    radars = build_final_radars(result["placements"], candidates, cand_ground_z, type_by_name)
    radars = assign_names(radars)

    print(f"\n[7] Exporting ...")
    write_placement_json(radars, OUTPUT_PLACEMENT)
    print(f"    Saved placement   -> {OUTPUT_PLACEMENT}  ({len(radars)} radars)")
    write_type_config_json(type_configs, OUTPUT_TYPE_CONFIG)
    print(f"    Saved type config -> {OUTPUT_TYPE_CONFIG}  ({len(type_configs)} types)")

    print(f"\n[Done]  {OUTPUT_PLACEMENT} is placement-only (no c_matrix.json emitted).")
    print(f"        Wiring this into build_matrix_1.3.py as an input source is future work.\n")


if __name__ == "__main__":
    main()
