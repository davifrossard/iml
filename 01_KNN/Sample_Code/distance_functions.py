from numpy import power, sqrt

def euclidean_distance(x1, x2):
    return sqrt(power((x1-x2),2).sum())