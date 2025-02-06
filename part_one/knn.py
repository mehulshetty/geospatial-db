### knn.py --- Finds the k closest POIs to a target POI using linear search

from helper import euclidean_distance
import pandas as pd

def knn_linear_search(dataset: pd.DataFrame, target_id: int, k: int):

    POI = dataset.loc[dataset['@id'] == target_id].iloc[0]

    dataset['@dist'] = dataset.apply(lambda row: euclidean_distance(row, POI), axis=1)

    top_k = dataset.nsmallest(k+1, '@dist', keep='first')

    result = top_k[1:][['@id', '@dist']].values.tolist()

    return result