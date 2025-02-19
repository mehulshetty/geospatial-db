### kd_tree.py --- Creates a KD-Tree Indes which is a binary search tree that 
# recursively partitions the space into half-spaces, adapting to the data distribution

import random
import pandas as pd
import time
from helper import euclidean_distance, plot_query

class Node:
    """
    Represents a node in the KD-Tree.
    """
    def __init__(self, poi=None, axis=None, left=None, right=None):
        self.poi = poi # POI data
        self.axis = axis    # Splitting axis (0 for latitude, 1 for longitude)
        self.left = left    # Left subtree
        self.right = right  # Right subtree

def build_kd_tree(dataset: pd.DataFrame):
    """
    Constructs a KD-Tree from the dataset.
    
    Parameters:
        dataset (pd.DataFrame): A DataFrame containing POI data with columns '@id', '@lat', '@lon', and 'name'.
    
    Returns:
        Node: The root node of the constructed KD-Tree.
    """
    points = dataset.to_dict('records')  # Convert DataFrame to list of dictionaries
    
    def build(points, depth=0):
        if not points:
            return None
        
        # Alternate between latitude (axis=0) and longitude (axis=1)
        axis = depth % 2
        points.sort(key=lambda p: p['@lat'] if axis == 0 else p['@lon'])
        
        # Select the median point as the current node
        median_idx = len(points) // 2
        median_point = points[median_idx]
        
        # Recursively build left and right subtrees
        left_subtree = build(points[:median_idx], depth + 1)
        right_subtree = build(points[median_idx + 1:], depth + 1)
        
        return Node(poi=median_point, axis=axis, left=left_subtree, right=right_subtree)
    
    return build(points)

def knn_kd_search(root: Node, target_poi: dict, k: int):
    """
    Finds the k-nearest neighbors to a target POI using KD-Tree search.
    
    Parameters:
        root (Node): The root node of the KD-Tree.
        target_poi (dict): The target POI with '@lat' and '@lon'.
        k (int): The number of nearest neighbors to find.
    
    Returns:
        list: A list of the '@id' and '@dist' of the k-nearest neighbors.
    """
    results = []  # List to store (distance, poi) tuples
    
    def search(node, depth=0):
        if node is None:
            return
        
        # Current point and distance
        point = node.poi
        dist = euclidean_distance(point, target_poi)
        
        # Add point to results
        results.append((dist, point))
        
        # Prune results to keep only top-k
        if len(results) > k:
            results.sort(key=lambda x: x[0])  # Sort by distance
            results.pop()  # Remove farthest point
        
        # Determine which subtree to explore first
        axis = node.axis
        target_val = target_poi['@lat'] if axis == 0 else target_poi['@lon']
        node_val = point['@lat'] if axis == 0 else point['@lon']
        
        if target_val < node_val:
            near_child = node.left
            far_child = node.right
        else:
            near_child = node.right
            far_child = node.left
        
        # Search the nearer subtree
        search(near_child, depth + 1)
        
        # Check if we need to search the farther subtree
        if len(results) < k or abs(target_val - node_val) < results[-1][0]:
            search(far_child, depth + 1)
    
    search(root)
    results.sort(key=lambda x: x[0])  # Final sort
    return [(p['@id'], d) for d, p in results]

def range_query_kd(root: Node, target_poi: dict, r: float):
    """
    Finds all POIs within r distance to a target POI using KD-Tree search.
    
    Parameters:
        root (KDNode): The root node of the KD-Tree.
        target_poi (dict): The target POI with '@lat' and '@lon'.
        r (float): The distance within which to find the POIs.
    
    Returns:
        list: A list of the '@id' and '@dist' of all nearby POIs.
    """
    results = []
    
    def search(node, depth=0):
        if node is None:
            return
        
        point = node.poi
        dist = euclidean_distance(point, target_poi)
        
        if dist <= r:
            results.append((point['@id'], dist))
        
        # Determine which subtree to explore first
        axis = node.axis
        target_val = target_poi['@lat'] if axis == 0 else target_poi['@lon']
        node_val = point['@lat'] if axis == 0 else point['@lon']
        
        if target_val < node_val:
            near_child = node.left
            far_child = node.right
        else:
            near_child = node.right
            far_child = node.left
        
        # Search the nearer subtree
        search(near_child, depth + 1)
        
        # Check if we need to search the farther subtree
        if abs(target_val - node_val) <= r:
            search(far_child, depth + 1)
    
    search(root)
    results.sort(key=lambda x: x[1])  # Sort by distance
    return results

def kd_tree_experiments(dataset: pd.DataFrame, config: dict):
    """
    Tests KD-Tree performance with different dataset sizes and query parameters.
    
    Parameters:
        dataset: Full POI dataset
        config: The set of configurations for the experiment
    
    Returns:
        Tuple of (knn_results, range_results) DataFrames
    """
    knn_results = []
    range_results = []
    
    for n in config["N_list"]:
        mini_df = dataset.iloc[:n].copy()
        
        # Build KD-Tree once per N
        start_build = time.perf_counter()
        root = build_kd_tree(mini_df)
        build_time = time.perf_counter() - start_build
        
        print(f"Built KD-Tree for N={n} in {build_time:.4f}s")
        
        # KNN Tests
        for k in config["k_list"]:
            target_id = random.choice(mini_df['@id'].values)
            target_poi = mini_df[mini_df['@id'] == target_id].iloc[0]
            
            start = time.perf_counter()
            knn_kd_search(root, target_poi, k)
            total_time = time.perf_counter() - start
            
            knn_results.append({
                'N': n,
                'k': k,
                'time': total_time
            })
        
        # Range Query Tests
        for r in config["r_list"]:
            target_id = random.choice(mini_df['@id'].values)
            target_poi = mini_df[mini_df['@id'] == target_id].iloc[0]

            start = time.perf_counter()
            range_query_kd(root, target_poi, r)
            total_time = time.perf_counter() - start
            
            range_results.append({
                'N': n,
                'r': r,
                'time': total_time
            })
    
    return pd.DataFrame(knn_results), pd.DataFrame(range_results)

def plot_kd(knn_results: pd.DataFrame, range_results: pd.DataFrame):

    print("HERE1")
    print(knn_results.head())
    print(range_results.head())

    plot_query(knn_results, 'k', "KD-Tree Index - kNN Query Performance", "k=")
    plot_query(range_results, 'r', "KD-Tree Index - Range Query Performance", "r=")

    return None