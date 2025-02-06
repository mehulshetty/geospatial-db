### helper.py --- Contains helper functions for the algorithms

import math

def euclidean_distance(point_x, point_y) -> float:
    x1, y1 = point_x['@lat'], point_x['@lon']
    x2, y2 = point_y['@lat'], point_y['@lon']

    distance = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    return distance