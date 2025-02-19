### brute_force.py --- Finds all POIs within a specified distance r from a target POI using linear search

from helper import euclidean_distance, plot_query
import pandas as pd
import matplotlib.pyplot as plt
import time
import random

def knn_linear_search(dataset: pd.DataFrame, target_id: int, k: int):
    """
    Finds the k-nearest neighbors to a target POI using linear search.

    Parameters:
        dataset (pd.DataFrame): A DataFrame containing POI data with columns '@id', '@lat', '@lon', and 'name'.
        target_id (int): The ID of the target POI.
        k (int): The number of nearest neighbors to find.

    Returns:
        list: A list of the '@id' and '@dist' of the k-nearest neighbors.
    """

    POI = dataset.loc[dataset['@id'] == target_id].iloc[0]

    dataset['@dist'] = dataset.apply(lambda row: euclidean_distance(row, POI), axis=1)

    top_k = dataset.nsmallest(k+1, '@dist', keep='first')

    result = top_k[1:][['@id', '@dist']].values.tolist()

    return result

def range_query_linear_search(dataset: pd.DataFrame, target_id: int, r: float):
    """
    Finds all POIs within r distance to a target POI using linear search.

    Parameters:
        dataset (pd.DataFrame): A DataFrame containing POI data with columns '@id', '@lat', '@lon', and 'name'.
        target_id (int): The ID of the target POI.
        r (float): The distance within which to find the POIs.

    Returns:
        list: A list of the '@id' and '@dist' of all nearby POIs.
    """

    POI = dataset.loc[dataset['@id'] == target_id].iloc[0]

    dataset['@dist'] = dataset.apply(lambda row: euclidean_distance(row, POI), axis=1)

    result = dataset.loc[(dataset['@dist'] > 0) & (dataset['@dist'] <= r)].values.tolist()

    return result

def brute_force_experiments(dataset: pd.DataFrame, config: dict):
    """
    Calculates the execution times of the kNN query for different dataset sizes (N) and k values.

    Parameters:
        dataset : The dataset for which we have to calculate execution times.

    Returns:
        pd.DataFrame: A DataFrame containing the execution times for each combination of dataset size and k.
    """
    knn_results = []
    range_results = []

    for n in config["N_list"]:
        mini_df = dataset.iloc[:n].copy()

        for k in config["k_list"]:
            target_id = random.choice(mini_df['@id'].values)

            start_time = time.perf_counter()
            knn_linear_search(mini_df, target_id, k)
            total_time = time.perf_counter() - start_time

            knn_results.append({
                'N': n,
                'k': k,
                'time': total_time
            })

        # Range Query Tests
        for r in config["r_list"]:
            target_id = random.choice(mini_df['@id'].values)
            
            start = time.perf_counter()
            range_query_linear_search(mini_df, target_id, r)
            total_time = time.perf_counter() - start
            
            range_results.append({
                'N': n,
                'r': r,
                'time': total_time
            })

    return pd.DataFrame(knn_results), pd.DataFrame(range_results)

# Plot the Data
def plot_brute_force(knn_results: pd.DataFrame, range_results: pd.DataFrame):

    plot_query(knn_results, 'k', "Brute Force - kNN Query Performance", "k=")
    plot_query(range_results, 'r', "Brute Force - Range Query Performance", "r=")

    return None