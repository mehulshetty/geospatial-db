### grid_index.py --- Builds a grid index which divides the space into uniform cells 
# and assigns each point to a cell based on its coordinates

from helper import check_bucket, euclidean_distance, expand_search_area, plot_query
import pandas as pd
import math
import pandas as pd
import matplotlib.pyplot as plt
import time
import random

def build_grid_index(dataset: pd.DataFrame, cell_size: float) -> dict:

    grid_index = {}

    for _, row in dataset.iterrows():
        key = check_bucket(row['@lat'], row['@lon'], cell_size)
        grid_index.setdefault(key, []).append(row)
    
    return grid_index

def knn_grid_search(dataset:pd.DataFrame, grid_index: dict, target_id: int, k: int, cell_size: float):
    """
    Finds the k-nearest neighbors to a target POI using grid search.

    Parameters:
        dataset (pd.DataFrame): A DataFrame containing POI data with columns '@id', '@lat', '@lon', and 'name'.
        target_id (int): The ID of the target POI.
        k (int): The number of nearest neighbors to find.
        cell_size (float): The size of each cell in the grid.

    Returns:
        list: A list of the '@id' and '@dist' of the k-nearest neighbors.
    """

    POI = dataset[dataset['@id'] == target_id].iloc[0]
    target_bucket = check_bucket(POI['@lat'], POI['@lon'], cell_size)

    filtered_buckets = []
    search_buckets = [target_bucket]

    while len(filtered_buckets) < k:
        current_buckets = expand_search_area(search_buckets)
        search_buckets = list(set(current_buckets))

        # Collect Candidates
        for bucket in search_buckets:
            filtered_buckets.extend(grid_index.get(bucket, []))

        # Filter And Sort Candidates
        candidates = [
            (p['@id'], euclidean_distance(p, POI)) 
            for p in filtered_buckets if p['@id'] != target_id
        ]
        candidates.sort(key=lambda x: x[1])

        if len(candidates) >= k:
            return candidates[:k]

    return candidates[:k]
        

def range_query_grid(dataset: pd.DataFrame, grid_index: dict, target_id: int, r: float, cell_size: float):
    """
    Finds all POIs within r distance to a target POI using grid search.

    Parameters:
        dataset (pd.DataFrame): A DataFrame containing POI data with columns '@id', '@lat', '@lon', and 'name'.
        target_id (int): The ID of the target POI.
        r (float): The distance within which to find the POIs.
        cell_size (float): The size of each cell in the grid.

    Returns:
        list: A list of the '@id' and '@dist' of all nearby POIs.
    """

    POI = dataset[dataset['@id'] == target_id].iloc[0]

    search_radius = math.ceil(r / cell_size)
    base_cell = check_bucket(POI['@lat'], POI['@lon'], cell_size)

    # Generate all cells to check
    cells_to_search = [
        (base_cell[0] + x, base_cell[1] + y)
        for x in range(-search_radius, search_radius+1)
        for y in range(-search_radius, search_radius+1)
    ]

    # Collect candidates
    candidates = []
    for cell in cells_to_search:
        candidates.extend(grid_index.get(cell, []))

    # Filter and sort
    results = []
    for poi in candidates:
        if poi['@id'] == target_id:
            continue
        dist = euclidean_distance(poi, POI)
        if dist <= r:
            results.append((poi['@id'], dist))
    
    return sorted(results, key=lambda x: x[1])

def grid_experiments(dataset, config):
    """
    Tests grid index performance with different cell sizes
    
    Parameters:
        dataset: Full POI dataset
        config: The set of configurations for the experiment
    
    Returns:
        Tuple of (knn_results, range_results) DataFrames
    """
    
    # Result storage
    knn_results = []
    range_results = []
    
    for cell_size in config["cell_sizes"]:
        
        for n in config["N_list"]:
            # Create subset
            mini_df = dataset.iloc[:min(n, len(dataset))].copy()
            
            # Build grid once per cell_size/N combination
            grid = build_grid_index(mini_df, cell_size)
            
            # KNN Tests
            for k in config["k_list"]:
                target_id = random.choice(mini_df['@id'].values)
                
                start = time.perf_counter()
                knn_grid_search(mini_df, grid, target_id, k, cell_size)
                total_time = time.perf_counter() - start
                
                knn_results.append({
                    'cell_size': cell_size,
                    'N': n,
                    'k': k,
                    'time': total_time
                })
            
            # Range Query Tests
            for r in config["r_list"]:
                target_id = random.choice(mini_df['@id'].values)
                
                start = time.perf_counter()
                range_query_grid(mini_df, grid, target_id, r, cell_size)
                total_time = time.perf_counter() - start
                
                range_results.append({
                    'cell_size': cell_size,
                    'N': n,
                    'r': r,
                    'time': total_time
                })
    
    return pd.DataFrame(knn_results), pd.DataFrame(range_results)

def plot_results(knn_results: pd.DataFrame, range_results: pd.DataFrame):

    plot_query(knn_results, 'cell_size', "Grid Index - KNN Query Performance", "Cell ")
    plot_query(range_results, 'cell_size', "Grid Index - Range Query Performance", "Cell ")

    return None



