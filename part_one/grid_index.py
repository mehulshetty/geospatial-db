### grid_index.py --- Builds a grid index which divides the space into uniform cells 
# and assigns each point to a cell based on its coordinates

from helper import check_bucket, euclidean_distance, expand_search_area
import pandas as pd

class grid_index:

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
        search_cells = [
            (target_bucket[0]+i, target_bucket[1]+j)
            for i in (-1,0,1)
            for j in (-1,0,1)
        ]

        while len(filtered_buckets) < k:
            temp_buckets = []

            for cell in search_cells:
                temp_buckets.extend(grid_index.get(cell, []))

                filtered_buckets = [
                    (t_POI['@id'], euclidean_distance(t_POI, POI))
                    for t_POI in temp_buckets
                    if t_POI['@id'] != target_id
                ]

            if len(filtered_buckets) < k:
                search_cells = expand_search_area(search_cells)
            else:
                break

        # Find top k nearest
        return sorted(filtered_buckets, key=lambda x: x[1])[:k]
    
    def range_query_grid(dataset: pd.DataFrame, target_id: int, r: float, cell_size: float):
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

        grid_index = grid_index.build_grid_index(dataset, cell_size)

        return None




