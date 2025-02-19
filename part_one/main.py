import pandas as pd

from grid_index import grid_experiments, plot_results as plot_grid
from kd_tree import kd_tree_experiments, plot_results as plot_kd

df = pd.read_csv('../dataset/clean_nyc_dataset.csv')

# Experiment Configuration
config = {
    'cell_sizes': [0.01, 0.05, 0.1, 0.2],
    'N_list': [1000, 10000, 100000, 825171],
    'k_list': [1, 5, 10, 50, 100, 500],
    'r_list': [0.01, 0.05, 0.1, 0.2, 0.5]
}

# Run experiments
knn_results_grid, range_results_grid = grid_experiments(df, config)
knn_results_kd, range_results_kd = kd_tree_experiments(df, config)

# Plot the Graph
plot_grid(knn_results_grid, range_results_grid)
plot_kd(knn_results_kd, range_results_kd)