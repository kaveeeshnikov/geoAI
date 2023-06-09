import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import os


PANORAMAS_PATH = ""    # path to panoramas
SAVE_GRIDS_PATH = ""   # path to the folder where the grids will be saved

# creates list of panoramas coordinates
coordinates_list = [os.path.splitext(filename)[0] for filename in os.listdir(PANORAMAS_PATH)]

# creating geodataframe from list of coordinates
locations_df = pd.DataFrame(coordinates_list, columns=["coordinate"])
locations_df[["lat", "lon"]] = (
    locations_df["coordinate"].str.split(",", expand=True).astype(float)
)
locations_gdf = gpd.GeoDataFrame(
    locations_df, geometry=gpd.points_from_xy(locations_df["lon"], locations_df["lat"])
)

# load earth map dataset from geopandas
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
world = world[world["continent"] != "Antarctica"]

def create_grids(geometry):
    """
    Creates grids on land or at coordinate points from dataframe.

    """
    cell_size = 2  # size of each cell in degrees
    x_min, y_min, x_max, y_max = world.total_bounds

    x_range = range(int(x_min), int(x_max), cell_size)
    y_range = range(int(y_min), int(y_max), cell_size)

    grid_cells = []
    for x in x_range:
        for y in y_range:
            cell = Polygon(
                [
                    (x, y),
                    (x + cell_size, y),
                    (x + cell_size, y + cell_size),
                    (x, y + cell_size),
                ]
            )
            if world.intersects(cell).any():
                grid_cells.append(cell)
            else:
                if cell.contains(geometry).any():
                    grid_cells.append(cell)
    
    return grid_cells

#create and save final geodataframe that used in model training
points = locations_gdf['geometry']
grid_cells_gdf = gpd.GeoDataFrame(geometry=create_grids(points))
locations_with_index_grid = gpd.sjoin(locations_gdf, grid_cells_gdf, how="left", rsuffix="grid")
locations_with_index_grid.rename(columns={"coordinate": "image"}, inplace=True)
locations_with_index_grid.to_csv(SAVE_GRIDS_PATH, index=False)
