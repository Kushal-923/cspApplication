"""
generate_czml.py
=================
Converts radar_meta.json (UTM/projected X,Y) + c_matrix.json (static LOS/FOV
coverage) + schedule_output.json (24-hour ON/OFF matrix from scheduler_1.2.py)
into a single CZML file that CesiumJS can play back on its timeline.

For every radar it now creates THREE kinds of CZML entities:
  1. A "site" entity    -> a point + label, always visible, marks the
     physical radar location.
  2. A "range" entity   -> a translucent PARTIAL ellipsoid (a sensor
     frustum, not a full sphere) whose azimuth/elevation extent is
     carved out with minimumClock/maximumClock/minimumCone/maximumCone,
     and whose `show` property is a list of time intervals that flip
     true/false to match that radar's hourly schedule.
  3. A handful of "ray" entities -> PolylineGraphics from the radar to a
     sample of the boundary points it actually covers (per c_matrix.json),
     time-dynamic on the same ON/OFF schedule as the range entity. These
     are the ground-truth LOS lines: because c_matrix.json was built by
     ray-marching against the real DEM, a ray drawn here to a "covered"
     point is a point that build_matrix_1.3.py already verified is
     terrain-visible from that radar. This sidesteps needing a live
     GPU LOS/shadow test in Cesium: the heavy lifting was already done
     offline, we're just drawing its result.

Coordinate handling
-------------------
radar_meta.json stores x/y in a projected CRS (UTM), matching the
TARGET_EPSG that build_matrix_1.3.py auto-detected from the border
file's centroid (see utm_epsg_for_lonlat() there). To stay consistent
with that pipeline instead of hardcoding a zone, this script repeats
the same auto-detection against punjab_border.geojson. If you already
know your EPSG code, set SOURCE_EPSG_OVERRIDE below to skip detection.

Boundary points (for LOS rays)
-------------------------------
c_matrix.json is an N x K binary matrix, but it does NOT carry the K
boundary points' coordinates — build_matrix_1.3.py only ever kept them
in memory. To draw a ray to "the boundary point c_matrix says radar i
covers", this script has to regenerate the same K points in the same
order, by repeating build_matrix_1.3.py's exact STEP 1 + STEP 2:
  - load + reproject the border to the auto-detected UTM EPSG
  - linemerge/union all segments into one LineString
  - interpolate every INTERVAL_M metres along it
INTERVAL_M here MUST match build_matrix_1.3.py's INTERVAL_M (250m by
default) or the regenerated points won't line up with c_matrix.json's
columns. If you ever change INTERVAL_M in build_matrix, change it here
too.

FOV bounds (elevation + azimuth)
---------------------------------
radar_meta.json does not carry theta_min/theta_max/az_halfwidth. This
script now loads radar_type_config.json (via --type-config, the same file
build_matrix_1.3.py/optimize_placement.py already produce and consume)
and looks up each radar's bounds by its "type" field -- there is no more
separate, hand-mirrored THETA_MIN_BY_TYPE/THETA_MAX_BY_TYPE/
AZIMUTH_HALFWIDTH_BY_TYPE table to keep in sync with build_matrix_1.3.py.
These bounds are only actually *enforced* in c_matrix.json if
ENABLE_FOV_CHECK / ENABLE_AZIMUTH_CHECK were True in build_matrix_1.3.py
when that file was generated; FOV_VISUAL_ENABLED / AZIMUTH_VISUAL_ENABLED
below are this script's own independent toggles for whether to draw the
loaded bounds at all. If radar_type_config.json is missing, or a radar's
type isn't a key in it, this script does NOT raise -- it prints a
warning and that radar's frustum simply renders as a full sphere instead
of a directional wedge, the same safe degrade as before (it just no
longer overstates precision it doesn't have, by design).

schedule_output.json — expected shape
--------------------------------------
This script accepts the raw dict that scheduler_1.2.py's `output`
variable would produce if you did:

    json.dump(output, open("schedule_output.json", "w"))

i.e. a dict containing an "schedule" key holding an N x M binary
matrix (N radars in the SAME ORDER as radar_meta.json, M time slots).
It also accepts a bare N x M list (no wrapping dict), and a
per-radar list of {"name": ..., "schedule": [...]} objects, in case
you post-processed the output differently. See `load_schedule()`.

Usage:
    pip install pyproj geopandas shapely --break-system-packages
    python generate_czml.py \
        --radar-meta radar_meta.json \
        --c-matrix c_matrix.json \
        --schedule schedule_output.json \
        --border punjab_border.geojson \
        --type-config radar_type_config.json \
        --out radar_coverage.czml
"""

import argparse
import colorsys
import json
import math
import sys
import zlib
from datetime import datetime, timedelta, timezone

from pyproj import Transformer
from shapely.ops import linemerge, unary_union

# Console output below uses an em-dash in a couple of warning messages --
# force UTF-8 on stdout so this doesn't crash under a plain Windows console
# (cp1252), which can't encode it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ══════════════════════════════════════════════════════════════
#  CONFIG — tweak colors / timing / styling here
# ══════════════════════════════════════════════════════════════

# Simulation start time (UTC). The 24 schedule slots are laid out as
# [START_TIME, START_TIME + 1h), [START_TIME + 1h, START_TIME + 2h), ...
# Change this to "today" if you want the Cesium timeline to open on
# the current date; a fixed date keeps runs reproducible.
START_TIME = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
SLOT_HOURS = 1.0   # must match NUM_TIME_SLOTS granularity in scheduler_1.2.py

# Set this to an integer EPSG code (e.g. 32643) to skip
# auto-detecting the UTM zone from the border file.
SOURCE_EPSG_OVERRIDE = None

# MUST match INTERVAL_M in build_matrix_1.3.py, or the boundary points
# regenerated here won't align with c_matrix.json's column order.
INTERVAL_M = 250

# Sphere/frustum color per radar type. Any arbitrary user-defined type name
# is supported (not just the original Small/Medium/Large): color_for_type()
# below derives a deterministic hue from a stable hash of the type name, so
# the same type name always gets the same color across runs, on any machine.
# A small override dict keeps the original three example types' familiar
# colors for visual continuity with existing screenshots/docs; anything
# else falls back to the hash-derived color. The override lookup is an
# EXACT match on the type name (never substring), unlike the old
# color_for_type()/BUDGET_BY_TYPE convention this replaces.
TYPE_COLOR_OVERRIDES = {
    "Small":  [66, 245, 194, 60],    # RGBA, alpha 0-255
    "Medium": [66, 165, 245, 60],
    "Large":  [245, 110, 66, 60],
}
COLOR_ALPHA     = 60   # alpha channel (0-255) for hash-derived colors, matching
                        # the translucent wedges above
POINT_COLOR     = [255, 255, 0, 255]
POINT_PIXELSIZE = 10
LABEL_FONT      = "13px sans-serif"

# Fraction of range_m used as the ellipsoid's innerRadii. Cesium's
# EllipsoidGeometry defaults innerRadii == radii, which makes the
# clock/cone "wall" and "cap" surfaces zero-area and invisible — the
# wedge then renders as only a thin outer shell. Setting innerRadii to
# a small non-zero fraction of range_m gives those walls real area,
# closing the wedge into a solid pie-slice volume from the radar origin
# out to the max range border.
INNER_RADIUS_FRACTION = 0.001

# ── FOV (elevation) / azimuth visualization ─────────────────────
# theta_min/theta_max/az_halfwidth are now loaded per-type from
# radar_type_config.json (see main()/build_czml()) -- no more hardcoded
# _BY_TYPE tables to keep in sync with build_matrix_1.3.py by hand.
FOV_VISUAL_ENABLED     = True   # flip True once bounds are confirmed & match c_matrix.json
AZIMUTH_VISUAL_ENABLED = True   # flip True once bounds are confirmed & match c_matrix.json

# ── LOS ray rendering ────────────────────────────────────────────
RAYS_ENABLED       = True
RAY_STRIDE         = 1     # draw every covered boundary point, no skipping
RAY_MAX_PER_RADAR  = 0      # 0 = no cap (thin_indices() already treats <=0 as unbounded);
                            # real coverage is small (max ~65 covered points for any one
                            # radar, ~813 total across all radars), so this is cheap
RAY_WIDTH          = 3.0
# Single constant color for every radar's LOS rays (not per-radar-type), so a ray is
# never the same hue as the coverage wedge it passes through.
LOS_RAY_COLOR      = [255, 0, 255, 255]

# ── Border coverage-point rendering ─────────────────────────────
BOUNDARY_COVERED_COLOR   = [144, 238, 144, 220]  # light green
BOUNDARY_UNCOVERED_COLOR = [255, 0, 0, 220]      # red
BOUNDARY_POINT_PIXELSIZE = 6

# ══════════════════════════════════════════════════════════════
#  CRS AUTO-DETECTION (mirrors build_matrix_1.3.py's logic so the
#  same UTM zone is used consistently across the whole pipeline)
# ══════════════════════════════════════════════════════════════

def utm_epsg_for_lonlat(lon, lat):
    zone = int(math.floor((lon + 180) / 6) + 1)
    return 32600 + zone if lat >= 0 else 32700 + zone


def detect_source_epsg(border_file):
    if SOURCE_EPSG_OVERRIDE:
        print(f"    Using SOURCE_EPSG_OVERRIDE = {SOURCE_EPSG_OVERRIDE}")
        return SOURCE_EPSG_OVERRIDE
    import geopandas as gpd
    gdf = gpd.read_file(border_file)
    wgs84 = gdf.to_crs(epsg=4326)
    c = wgs84.geometry.union_all().centroid
    epsg = utm_epsg_for_lonlat(c.x, c.y)
    print(f"    Auto-detected UTM zone from {border_file}: EPSG:{epsg}")
    return epsg


def regenerate_boundary_points(border_file, target_epsg, interval_m):
    """
    Reproduces build_matrix_1.3.py STEP 1 + STEP 2 exactly, so the
    resulting list of (x, y) points lines up 1:1 with c_matrix.json's
    columns. Returns a list of (x, y) tuples in target_epsg.
    """
    import geopandas as gpd
    border_gdf = gpd.read_file(border_file).to_crs(epsg=target_epsg)
    line = linemerge(unary_union(border_gdf.geometry))
    if line.geom_type != "LineString":
        raise RuntimeError(
            f"linemerge produced {line.geom_type} instead of LineString — "
            "boundary points can't be regenerated to match c_matrix.json. "
            "Inspect the border geojson for discontinuities."
        )
    n_pts = int(line.length // interval_m)
    pts = [line.interpolate(i * interval_m) for i in range(n_pts + 1)]
    return [(p.x, p.y) for p in pts]


# ══════════════════════════════════════════════════════════════
#  SCHEDULE LOADING — tolerant of a few reasonable output shapes
# ══════════════════════════════════════════════════════════════

def load_schedule(schedule_file, radar_names):
    """
    Returns a dict: {radar_name: [0/1, 0/1, ...]} aligned to radar_names.
    Accepts:
      (a) {"schedule": [[...]]}                      <- raw scheduler_1.2 output
      (b) [[...]]                                     <- bare N x M matrix
      (c) [{"name": "...", "schedule": [...]}]        <- per-radar records
    """
    with open(schedule_file) as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "schedule" in raw:
        matrix = raw["schedule"]
        return {name: matrix[i] for i, name in enumerate(radar_names)}

    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        return {name: raw[i] for i, name in enumerate(radar_names)}

    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        by_name = {}
        for rec in raw:
            key = rec.get("name") or rec.get("radar") or rec.get("id")
            by_name[key] = rec.get("schedule") or rec.get("on") or rec.get("values")
        return {name: by_name.get(name) for name in radar_names}

    raise ValueError(
        "Unrecognized schedule_output.json shape — see load_schedule() docstring."
    )


def slots_to_intervals(on_off, start_time, slot_hours):
    """
    Collapse a list of 0/1 per-slot values into merged (iso_start, iso_end, bool)
    runs, so we emit one CZML interval per contiguous run of ON or OFF instead
    of one per hour. Also returns the full-day availability string.
    """
    intervals = []
    n = len(on_off)
    i = 0
    while i < n:
        val = bool(on_off[i])
        j = i
        while j < n and bool(on_off[j]) == val:
            j += 1
        t0 = start_time + timedelta(hours=slot_hours * i)
        t1 = start_time + timedelta(hours=slot_hours * j)
        intervals.append((iso(t0), iso(t1), val))
        i = j
    return intervals


def iso(dt):
    """ISO 8601 with a trailing 'Z', the format CZML expects."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ══════════════════════════════════════════════════════════════
#  ANGLE MATH — degrees (radar convention) -> Cesium clock/cone radians
#  See the chat response for the full derivation; short version here:
#
#  Cesium's EllipsoidGraphics minimumClock/maximumClock/minimumCone/
#  maximumCone describe a wedge of a sphere in the entity's LOCAL
#  East-North-Up (ENU) frame (Cesium builds that frame automatically
#  from a cartographicDegrees position — no orientation quaternion
#  needed):
#    - "cone" is measured from the local +z (zenith/straight up) axis,
#      0 at zenith, pi/2 at the local horizon, pi at nadir (straight down).
#    - "clock" is measured in the local x-y (East-North) plane from the
#      local +x (East) axis, increasing counter-clockwise toward +y
#      (North) — i.e. the exact same atan2(dy, dx) convention
#      build_matrix_1.3.py already uses for boresight_deg and in_azimuth().
#
#  Elevation (theta, degrees above horizontal, build_matrix_1.3.py's
#  in_fov() convention) -> cone (radians from zenith):
#      cone = radians(90 - theta)
#  So theta_max (looking up) gives the SMALLEST cone (closest to zenith)
#  -> minimumCone, and theta_min (looking down) gives the LARGEST cone
#  -> maximumCone.
#
#  Azimuth (boresight_deg +/- az_halfwidth, same atan2 convention as
#  Cesium's clock) needs no axis conversion at all, just deg -> rad:
#      minimumClock = radians(boresight_deg - az_halfwidth)
#      maximumClock = radians(boresight_deg + az_halfwidth)
# ══════════════════════════════════════════════════════════════

def cone_bounds_rad(theta_min_deg, theta_max_deg):
    min_cone = math.radians(90.0 - theta_max_deg)
    max_cone = math.radians(90.0 - theta_min_deg)
    return min_cone, max_cone


def clock_bounds_rad(boresight_deg, halfwidth_deg):
    return (
        math.radians(boresight_deg - halfwidth_deg),
        math.radians(boresight_deg + halfwidth_deg),
    )


# ══════════════════════════════════════════════════════════════
#  CZML BUILDING
# ══════════════════════════════════════════════════════════════

def color_for_type(radar_type):
    """
    Exact-match override for the three original example types (visual
    continuity); everything else gets a hash-derived hue. Uses
    zlib.crc32, NOT builtin hash() -- Python's hash() on str is randomized
    per-process (PYTHONHASHSEED) unless disabled, so it would give a
    different color every run for the same type name.
    """
    if radar_type in TYPE_COLOR_OVERRIDES:
        return TYPE_COLOR_OVERRIDES[radar_type]
    digest = zlib.crc32(radar_type.encode("utf-8"))
    hue = (digest % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return [int(round(r * 255)), int(round(g * 255)), int(round(b * 255)), COLOR_ALPHA]


def thin_indices(indices, stride, max_count):
    """Take every `stride`-th covered index, then cap the total count,
    re-spacing the cap evenly across the remaining list rather than just
    truncating (so a hard cap doesn't just show the first stretch of border)."""
    strided = indices[::stride] if stride > 1 else list(indices)
    if len(strided) <= max_count or max_count <= 0:
        return strided
    step = len(strided) / max_count
    return [strided[int(i * step)] for i in range(max_count)]


def border_point_color_intervals(j, c_matrix, radars, schedules, day_start, day_end, slot_hours):
    """Dynamic green/red CZML color-interval list for boundary point `j`: green
    whenever at least one radar that covers it (per c_matrix) is scheduled ON,
    red otherwise. Reuses slots_to_intervals() for the run-merging instead of
    reimplementing it — same pattern as the per-radar `show_intervals`."""
    covering = [i for i, row in enumerate(c_matrix) if row[j]]
    if not covering:
        return [{"interval": f"{iso(day_start)}/{iso(day_end)}", "rgba": BOUNDARY_UNCOVERED_COLOR}]

    num_slots = len(next(s for s in schedules.values() if s))
    combined = [0] * num_slots
    for i in covering:
        on_off = schedules.get(radars[i]["name"]) or [0] * num_slots
        combined = [a or b for a, b in zip(combined, on_off)]

    merged = slots_to_intervals(combined, day_start, slot_hours)
    return [
        {
            "interval": f"{t0}/{t1}",
            "rgba": BOUNDARY_COVERED_COLOR if val else BOUNDARY_UNCOVERED_COLOR,
        }
        for (t0, t1, val) in merged
    ]


def build_czml(radars, schedules, c_matrix, boundary_lonlat, day_start, day_end, type_config):
    doc_availability = f"{iso(day_start)}/{iso(day_end)}"

    czml = [
        {
            "id": "document",
            "name": "Radar Coverage Simulation",
            "version": "1.0",
            "clock": {
                "interval": doc_availability,
                "currentTime": iso(day_start),
                "multiplier": 120,     # 120x real time; tweak playback speed here
                "range": "LOOP_STOP",
                "step": "SYSTEM_CLOCK_MULTIPLIER",
            },
        }
    ]

    for radar_index, r in enumerate(radars):
        name = r["name"]
        lon, lat = r["lon"], r["lat"]
        ground_z = r["ground_z"]
        antenna_z = r["z"]          # ground_z + h_ant
        range_m = r["range_m"]
        rgba = color_for_type(r["type"])

        # ---- Entity 1: radar site marker (point + label), always shown ----
        czml.append({
            "id": f"{name}_site",
            "name": name,
            "availability": doc_availability,
            "position": {
                "cartographicDegrees": [lon, lat, ground_z]
            },
            "properties": {
                "type": r["type"],
                "range_m": r["range_m"],
                "boresight_deg": r.get("boresight_deg"),
                "antennaHeight": r["h_ant"],
            },
            "point": {
                "pixelSize": POINT_PIXELSIZE,
                "color": {"rgba": POINT_COLOR},
                "outlineColor": {"rgba": [0, 0, 0, 255]},
                "outlineWidth": 1,
                "heightReference": "CLAMP_TO_GROUND",
            },
            "label": {
                "text": name,
                "font": LABEL_FONT,
                "fillColor": {"rgba": [255, 255, 255, 255]},
                "outlineColor": {"rgba": [0, 0, 0, 255]},
                "outlineWidth": 2,
                "style": "FILL_AND_OUTLINE",
                "verticalOrigin": "BOTTOM",
                "pixelOffset": {"cartesian2": [0, -12]},
                "heightReference": "CLAMP_TO_GROUND",
            },
        })

        # ---- Compute this radar's ON/OFF show intervals (shared by the
        #      frustum AND its LOS rays — both should only be visible
        #      while the radar is actually scheduled ON) ----
        on_off = schedules.get(name)
        if on_off is None:
            print(f"    WARNING: no schedule found for {name} — frustum/rays default to always OFF")
            show_intervals = [{"interval": doc_availability, "boolean": False}]
        else:
            merged = slots_to_intervals(on_off, day_start, SLOT_HOURS)
            show_intervals = [
                {"interval": f"{t0}/{t1}", "boolean": val} for (t0, t1, val) in merged
            ]

        # ---- Entity 2: coverage frustum (partial ellipsoid), visibility
        #      driven by schedule ----
        inner_r = max(1.0, range_m * INNER_RADIUS_FRACTION)
        ellipsoid_packet = {
            "radii": {"cartesian": [range_m, range_m, range_m]},
            "innerRadii": {"cartesian": [inner_r, inner_r, inner_r]},
            "fill": True,
            "material": {
                "solidColor": {
                    "color": {"rgba": rgba}
                }
            },
            "outline": True,
            "outlineColor": {"rgba": [rgba[0], rgba[1], rgba[2], 180]},
            "slicePartitions": 24,
            "stackPartitions": 12,
            "show": show_intervals,
        }

        if FOV_VISUAL_ENABLED:
            r_type_cfg = type_config.get(r["type"], {})
            theta_min = r_type_cfg.get("theta_min")
            theta_max = r_type_cfg.get("theta_max")
            if theta_min is not None and theta_max is not None:
                min_cone, max_cone = cone_bounds_rad(theta_min, theta_max)
                ellipsoid_packet["minimumCone"] = min_cone
                ellipsoid_packet["maximumCone"] = max_cone

        if AZIMUTH_VISUAL_ENABLED:
            r_type_cfg = type_config.get(r["type"], {})
            az_halfwidth = r_type_cfg.get("az_halfwidth")
            boresight_deg = r.get("boresight_deg")
            if az_halfwidth is not None and boresight_deg is not None:
                min_clock, max_clock = clock_bounds_rad(boresight_deg, az_halfwidth)
                ellipsoid_packet["minimumClock"] = min_clock
                ellipsoid_packet["maximumClock"] = max_clock

        czml.append({
            "id": f"{name}_range",
            "name": f"{name} coverage range",
            "availability": doc_availability,
            # NOTE ON HEIGHT: `antenna_z` here is a DEM-derived (orthometric,
            # geoid-referenced) height, but CZML/Cesium treats an unqualified
            # cartographicDegrees height as ELLIPSOIDAL. EllipsoidGraphics also
            # has no heightReference property, so we can't CLAMP_TO_GROUND it
            # here. This position is therefore only a rough fallback — the
            # frontend (index.html) re-positions this entity after sampling
            # the real rendered terrain, using the "antennaHeight" property
            # below plus the sampled terrain height. Do not rely on this
            # baked-in height for anything precision-sensitive.
            "position": {
                "cartographicDegrees": [lon, lat, antenna_z]
            },
            "properties": {
                "antennaHeight": r["h_ant"],
            },
            "ellipsoid": ellipsoid_packet,
        })

        # ---- Entity 3+: LOS rays to a sample of covered boundary points ----
        if RAYS_ENABLED and c_matrix is not None and boundary_lonlat is not None:
            row = c_matrix[radar_index] if radar_index < len(c_matrix) else []
            covered = [j for j, v in enumerate(row) if v]
            chosen = thin_indices(covered, RAY_STRIDE, RAY_MAX_PER_RADAR)

            for j in chosen:
                if j >= len(boundary_lonlat):
                    continue
                pt_lon, pt_lat = boundary_lonlat[j]
                czml.append({
                    "id": f"{name}_ray_{j}",
                    "name": f"{name} LOS to boundary pt {j}",
                    "availability": doc_availability,
                    # Rough baked heights (radar end = antenna_z, target end
                    # = this radar's own ground_z as a local approximation —
                    # terrain rarely changes drastically over one radar's
                    # range). index.html corrects BOTH endpoints precisely
                    # by sampling actual rendered terrain height, same as it
                    # does for the "_range" ellipsoid above.
                    "polyline": {
                        "positions": {
                            "cartographicDegrees": [
                                lon, lat, antenna_z,
                                pt_lon, pt_lat, ground_z,
                            ]
                        },
                        "width": RAY_WIDTH,
                        "material": {
                            "solidColor": {
                                "color": {"rgba": LOS_RAY_COLOR}
                            }
                        },
                        # Material used for the portions of the line that fail the depth
                        # test (i.e. occluded by/passing through the translucent coverage
                        # ellipsoid or terrain) — without this, those portions simply
                        # wouldn't draw. Same color, so the ray stays visibly magenta
                        # whether it's in front of or behind other geometry.
                        "depthFailMaterial": {
                            "solidColor": {
                                "color": {"rgba": LOS_RAY_COLOR}
                            }
                        },
                        "show": show_intervals,
                    },
                    "properties": {
                        "antennaHeight": r["h_ant"],
                        "targetHeight": 0,
                    },
                })

    # ---- Border coverage points: one per boundary point used by c_matrix.json,
    #      colored green while >=1 covering radar is scheduled ON, red otherwise.
    #      Replaces the flat yellow border with a live coverage-status view. ----
    if c_matrix is not None and boundary_lonlat is not None:
        for j, (lon, lat) in enumerate(boundary_lonlat):
            czml.append({
                "id": f"boundary_pt_{j}",
                "availability": doc_availability,
                # Nominal height — CLAMP_TO_GROUND (same as the "_site" markers
                # above) snaps this to whatever terrain Cesium actually renders,
                # so no terrain-sampling correction is needed in index.html.
                "position": {"cartographicDegrees": [lon, lat, 0]},
                "point": {
                    "pixelSize": BOUNDARY_POINT_PIXELSIZE,
                    "color": border_point_color_intervals(
                        j, c_matrix, radars, schedules, day_start, day_end, SLOT_HOURS
                    ),
                    "outlineColor": {"rgba": [0, 0, 0, 200]},
                    "outlineWidth": 1,
                    "heightReference": "CLAMP_TO_GROUND",
                },
            })

    return czml


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radar-meta", default="radar_meta.json")
    ap.add_argument("--c-matrix", default="c_matrix.json")
    ap.add_argument("--schedule", default="schedule_output.json")
    ap.add_argument("--border", default="punjab_border.geojson")
    ap.add_argument("--type-config", default="radar_type_config.json",
                     help="Per-type FOV/azimuth bounds (theta_min/theta_max/az_halfwidth), "
                          "produced by optimize_placement.py. Missing file is a non-fatal "
                          "warning -- the coverage wedge degrades to a full sphere.")
    ap.add_argument("--out", default="radar_coverage.czml")
    ap.add_argument("--no-rays", action="store_true", help="Skip LOS ray polylines entirely.")
    args = ap.parse_args()

    global RAYS_ENABLED
    if args.no_rays:
        RAYS_ENABLED = False

    print(f"[1] Detecting source CRS from {args.border} ...")
    src_epsg = detect_source_epsg(args.border)
    transformer = Transformer.from_crs(f"EPSG:{src_epsg}", "EPSG:4326", always_xy=True)

    print(f"[2] Loading {args.radar_meta} ...")
    with open(args.radar_meta) as f:
        meta = json.load(f)

    radars = []
    for r in meta:
        lon, lat = transformer.transform(r["x"], r["y"])
        radars.append({
            "name": r["name"],
            "lon": lon,
            "lat": lat,
            "ground_z": r["ground_z"],
            "z": r["z"],
            "h_ant": r["h_ant"],
            "range_m": r["range_m"],
            "type": r["type"],
            "boresight_deg": r.get("boresight_deg"),
        })
    print(f"    Converted {len(radars)} radar positions to WGS84 lon/lat")

    print(f"[3] Loading {args.type_config} ...")
    try:
        with open(args.type_config) as f:
            type_config = json.load(f)
        if not isinstance(type_config, dict):
            print(f"    WARNING: {args.type_config} is not a JSON object (got "
                  f"{type(type_config).__name__}) -- ignoring it. FOV/azimuth "
                  "wedges will render as full spheres.")
            type_config = {}
        else:
            print(f"    Loaded config for {len(type_config)} type(s): {sorted(type_config.keys())}")
    except FileNotFoundError:
        print(f"    WARNING: {args.type_config} not found -- proceeding without "
              "per-type FOV/azimuth bounds. Coverage wedges will render as full "
              "spheres instead of directional sectors (this is the same safe "
              "degrade the old hardcoded THETA_MIN_BY_TYPE.get() lookup used "
              "to fall back to for an unrecognized type).")
        type_config = {}

    print(f"[4] Loading {args.schedule} ...")
    schedules = load_schedule(args.schedule, [r["name"] for r in meta])
    num_slots = len(next((s for s in schedules.values() if s), []))
    day_end = START_TIME + timedelta(hours=SLOT_HOURS * num_slots)
    print(f"    {num_slots} time slots -> window {iso(START_TIME)} to {iso(day_end)}")

    # c_matrix/boundary points are needed for both the LOS rays (gated by
    # RAYS_ENABLED below) AND the border coverage points (always drawn), so
    # this load is unconditional — --no-rays only suppresses the ray
    # polylines themselves, inside build_czml().
    c_matrix = None
    boundary_lonlat = None
    try:
        print(f"[5] Loading {args.c_matrix} ...")
        with open(args.c_matrix) as f:
            c_matrix = json.load(f)
        print(f"    Loaded {len(c_matrix)} x {len(c_matrix[0]) if c_matrix else 0} coverage matrix")

        print(f"[6] Regenerating boundary points from {args.border} "
              f"(INTERVAL_M={INTERVAL_M}) to align with c_matrix.json columns ...")
        boundary_xy = regenerate_boundary_points(args.border, src_epsg, INTERVAL_M)
        if len(boundary_xy) != len(c_matrix[0]):
            print(f"    WARNING: regenerated {len(boundary_xy)} boundary points but "
                  f"c_matrix.json has {len(c_matrix[0])} columns — INTERVAL_M or the "
                  "border file probably don't match what build_matrix_1.3.py used. "
                  "Rays/border coloring may be misaligned or skipped.")
        boundary_lonlat = [transformer.transform(x, y) for x, y in boundary_xy]
        print(f"    Regenerated {len(boundary_lonlat)} boundary points")
    except FileNotFoundError:
        print(f"    WARNING: {args.c_matrix} not found — skipping LOS rays and "
              "border coverage coloring (frustum/schedule visualization still works without it).")

    print(f"[7] Building CZML ...")
    czml = build_czml(radars, schedules, c_matrix, boundary_lonlat, START_TIME, day_end, type_config)

    with open(args.out, "w") as f:
        json.dump(czml, f, indent=2)
    n_rays = sum(1 for p in czml if "_ray_" in p.get("id", ""))
    n_boundary_pts = sum(1 for p in czml if p.get("id", "").startswith("boundary_pt_"))
    print(f"\n[Done] Wrote {args.out}  ({len(czml)} packets, {len(radars)} radars, "
          f"{n_rays} LOS rays, {n_boundary_pts} border coverage points)")
    print(f"       Point the frontend's CZML_URL at this file.")


if __name__ == "__main__":
    main()
