### range_query.py --- Finds all POIs within a specified distance r from a target POI using linear search

from helper import euclidean_distance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random

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

def calculate_execution_time(N_list: list, r_list: list, dataset: pd.DataFrame):
    """
    Calculates the execution times of the kNN query for different dataset sizes (N) and r values.

    Parameters:
        N_list : List of sizes for the dataset.
        r_list : A list of r values to test.
        dataset : The dataset for which we have to calculate execution times.

    Returns:
        pd.DataFrame: A DataFrame containing the execution times for each combination of dataset size and r.
    """

    results = {r: [] for r in r_list}

    for n in N_list:
        mini_df = dataset[:min(n, len(dataset))].copy()

        for r in r_list:
            target_id = random.choice(mini_df['@id'].values)

            start_time = time.time()
            range_query_linear_search(mini_df, target_id, r)
            total_time = time.time() - start_time

            results[r].append(total_time)

    return results

# Plot the Data

df = pd.read_csv('../dataset/clean_nyc_dataset.csv')

r_list = [0.01, 0.05, 0.1, 0.2, 0.5]
N_list = [1000, 10000, 100000, 825171]

plot_data = calculate_execution_time(N_list, r_list, df)

for r, exec_time in plot_data.items():
    plt.plot(N_list, exec_time, label=f'r = {r}', marker='o')

plt.title('Execution Time vs Dataset Size (N) for Different r Values')
plt.xlabel('Dataset Size (N)')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True, linestyle='--')
plt.show()