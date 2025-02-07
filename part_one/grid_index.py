### grid_index.py --- Builds a grid index which divides the space into uniform cells 
# and assigns each point to a cell based on its coordinates

from helper import check_bucket
import pandas as pd

def build_grid_index(dataset: pd.DataFrame, cell_size: float) -> dict:

    grid_index = {}

    for _, row in dataset.iterrows():
        key = check_bucket(row['@lat'], row['@lon'], cell_size)
        grid_index.setdefault(key, []).append(row)
    
    return grid_index



