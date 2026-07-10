"""
get_dem.py — Download a Copernicus GLO-30 (COP30) DEM tile clipped to
your Punjab border AOI, via the OpenTopography API.

What this does:
  1. Reads punjab_border.geojson and finds its bounding box.
  2. Pads that box with a safety margin = INLAND_M + max radar range,
     so every possible radar position (up to 2000m inland + up to 15km
     range) is guaranteed to fall inside DEM coverage. This directly
     avoids the "out of bounds" RuntimeError we added to DEMReader.
  3. Calls the OpenTopography COP30 API with that padded box.
  4. Saves the result as punjab_dem.tif — matches the DEM_FILE default
     in build_matrix_1_3.py, so you can point straight at it.

Before running:
  pip install requests geopandas shapely --break-system-packages
  Get a free API key: https://opentopography.org -> MyOpenTopo Dashboard
                       -> "Request API Key"

Usage:
  python get_dem.py YOUR_API_KEY_HERE
"""

import sys
import math
import requests
import geopandas as gpd

BORDER_FILE = "punjab_border.geojson"
OUTPUT_DEM  = "punjab_dem.tif"

# Must match build_matrix_1_3.py's config — this is exactly the margin
# the DEMReader bounds check would otherwise reject if it were too small.
INLAND_M       = 2000
MAX_RADAR_M    = 15_000   # largest range_m in RADAR_SPECS (Large radars)
SAFETY_PAD_M   = INLAND_M + MAX_RADAR_M   # 17,000 m
EXTRA_MARGIN_M = 2_000     # a bit more slack, DEM is free, better safe

TOTAL_MARGIN_M = SAFETY_PAD_M + EXTRA_MARGIN_M   # 19,000 m ≈ 19 km


def main():
    if len(sys.argv) != 2:
        print("Usage: python get_dem.py YOUR_API_KEY_HERE")
        sys.exit(1)
    api_key = sys.argv[1]

    print(f"[1] Reading {BORDER_FILE} for bounding box ...")
    gdf = gpd.read_file(BORDER_FILE)
    gdf_wgs84 = gdf.to_crs(epsg=4326)   # OpenTopography wants lat/lon
    minx, miny, maxx, maxy = gdf_wgs84.total_bounds
    print(f"    Raw border bbox (lon/lat): "
          f"W={minx:.4f} E={maxx:.4f} S={miny:.4f} N={maxy:.4f}")

    # Convert metres margin to degrees. Longitude degrees shrink with
    # latitude (~cos(lat)); latitude degrees are ~constant (~111km/deg).
    # Using the border's mean latitude for the longitude correction.
    mean_lat = (miny + maxy) / 2.0
    deg_per_m_lat = 1.0 / 111_320.0
    deg_per_m_lon = 1.0 / (111_320.0 * math.cos(math.radians(mean_lat)))

    pad_lat = TOTAL_MARGIN_M * deg_per_m_lat
    pad_lon = TOTAL_MARGIN_M * deg_per_m_lon

    south = miny - pad_lat
    north = maxy + pad_lat
    west  = minx - pad_lon
    east  = maxx + pad_lon

    print(f"    Margin applied  : {TOTAL_MARGIN_M/1000:.1f} km "
          f"(inland {INLAND_M}m + max range {MAX_RADAR_M/1000:.0f}km + slack)")
    print(f"    Padded bbox     : W={west:.4f} E={east:.4f} S={south:.4f} N={north:.4f}")

    area_km2 = (north - south) * 111.32 * (east - west) * 111.32 * math.cos(math.radians(mean_lat))
    print(f"    Approx area     : {area_km2:,.0f} km^2  (COP30 limit: 450,000 km^2)")

    print(f"\n[2] Requesting COP30 DEM from OpenTopography ...")
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": "COP30",
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    resp = requests.get(url, params=params, timeout=120)

    if resp.status_code != 200:
        print(f"    ERROR {resp.status_code}: {resp.text[:500]}")
        sys.exit(1)

    with open(OUTPUT_DEM, "wb") as f:
        f.write(resp.content)

    print(f"    Saved -> {OUTPUT_DEM}  ({len(resp.content)/1e6:.1f} MB)")
    print(f"\n[Done] Point DEM_FILE = \"{OUTPUT_DEM}\" in build_matrix_1_3.py — "
          f"no reprojection needed, DEMReader handles that automatically.")


if __name__ == "__main__":
    main()
