### main.py --- 
import pandas as pd

from brute_force import brute_force_experiments, plot_brute_force
from grid_index import grid_experiments, plot_grid
from kd_tree import kd_tree_experiments, plot_kd

df = pd.read_csv('../dataset/clean_nyc_dataset.csv')

# Experiment Configuration
config = {
    'cell_sizes': [0.01, 0.05, 0.1, 0.2],
    'N_list': [1000, 10000, 100000, 825171],
    'k_list': [1, 5, 10, 50, 100, 500],
    'r_list': [0.01, 0.05, 0.1, 0.2, 0.5]
}

# Run experiments
print("Running Brute Force Experiments")
#knn_results_brute_force, range_results_brute_force = brute_force_experiments(df, config)
print("Running Grid Index Experiments")
knn_results_grid, range_results_grid = grid_experiments(df, config)
print("Running KD-Tree Index Experiments")
#knn_results_kd, range_results_kd = kd_tree_experiments(df, config)

# Plot the Graph
print("Plotting Brute Force")
#plot_brute_force(knn_results_brute_force, range_results_brute_force)
print("Plotting Grid")
plot_grid(knn_results_grid, range_results_grid)
print("Plotting KD-Tree")
#plot_kd(knn_results_kd, range_results_kd)