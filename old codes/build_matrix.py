import geopandas as gpd
from shapely.geometry import Point
import rasterio
import math
import json

print("1. Loading Border Data...")
border_gdf = gpd.read_file('punjab_border.geojson')
border_gdf = border_gdf.to_crs(epsg=32643)
line = border_gdf.geometry.iloc[0]

print("2. Discretizing the Border (50m intervals)...")
interval = 50
num_points = int(line.length // interval)
boundary_points = [line.interpolate(i * interval) for i in range(num_points + 1)]
print(f"Generated {len(boundary_points)} boundary points.")

print("3. Deploying Radar Network...")
# Let's space 5 radars out along the border (at 10%, 30%, 50%, 70%, 90% marks)
# We will place them 2 km (2000m) inland.
# Form: [Name, placement_percentage, range_in_meters]
radar_specs = [
    {"name": "Radar 1 (Small)", "pos_pct": 0.10, "range": 5000},
    {"name": "Radar 2 (Medium)", "pos_pct": 0.30, "range": 10000},
    {"name": "Radar 3 (Large)", "pos_pct": 0.50, "range": 15000},
    {"name": "Radar 4 (Small)", "pos_pct": 0.70, "range": 5000},
    {"name": "Radar 5 (Medium)", "pos_pct": 0.90, "range": 10000}
]

radars = []
for spec in radar_specs:
    # Find the coordinate on the line based on the percentage
    target_idx = int(len(boundary_points) * spec["pos_pct"])
    base_point = boundary_points[target_idx]
    
    # Shift 2000 meters inland (adjusting X coordinate)
    radar_coords = (base_point.x + 2000, base_point.y)
    radars.append({
        "name": spec["name"],
        "coords": radar_coords,
        "range": spec["range"]
    })
    print(f"Deployed {spec['name']} with {spec['range']/1000}km range.")

print("\n4. Generating the 2D C_ij Coverage Matrix...")
C_matrix = [] # This will hold lists of 1s and 0s

with rasterio.open('punjab_dem.tif') as dem:
    
    # Loop through each radar (Rows)
    for i, radar in enumerate(radars):
        radar_coverage_row = []
        
        # Loop through every border point (Columns)
        for point in boundary_points:
            point_coords = (point.x, point.y)
            distance = math.dist(radar["coords"], point_coords)
            
            if distance <= radar["range"]:
                radar_coverage_row.append(1)
            else:
                radar_coverage_row.append(0)
                
        C_matrix.append(radar_coverage_row)
        covered_count = sum(radar_coverage_row)
        print(f"Row {i} ({radar['name']}): Covers {covered_count} / {len(boundary_points)} points.")

# Export the matrix so OR-Tools can use it in the next script
with open('c_matrix.json', 'w') as f:
    json.dump(C_matrix, f)

print("\nSuccess! C_matrix generated and saved to 'c_matrix.json'.")