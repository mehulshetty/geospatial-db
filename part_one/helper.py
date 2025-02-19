### helper.py --- Contains helper functions for the algorithms

import math
import pandas as pd
import matplotlib.pyplot as plt

def euclidean_distance(point_x, point_y) -> float:
    x1, y1 = point_x['@lat'], point_x['@lon']
    x2, y2 = point_y['@lat'], point_y['@lon']

    distance = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    return distance

def check_bucket(lat, lon, cell_size) -> tuple:
    lat_bucket = round(lat / cell_size)
    lon_bucket = round(lon / cell_size)
    return (lat_bucket, lon_bucket)

def expand_search_area(cells):
    # Implementation for dynamic grid expansion
    min_x = min(c[0] for c in cells)
    max_x = max(c[0] for c in cells)
    min_y = min(c[1] for c in cells)
    max_y = max(c[1] for c in cells)
    
    return [
        (x, y) 
        for x in range(min_x-1, max_x+2)
        for y in range(min_y-1, max_y+2)
    ]

def plot_query(data: pd.DataFrame, param: str, title: str, label_prefix: str):
        """Helper function to plot query results with common formatting"""
        plt.figure(figsize=(12, 6))
        linestyle = '--' if param in ['k', 'r'] else '-'
        
        for value in data[param].unique():
            subset = data[data[param] == value]
            plt.plot(subset['N'], subset['time'], 
                     marker='o', linestyle=linestyle,
                     label=f'{label_prefix}{value})' if param in ['k', 'r'] else f'{label_prefix}{value}')
        
        plt.title(title)
        plt.xlabel("Dataset Size (N)")
        plt.ylabel("Query Execution Time (s)")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()
