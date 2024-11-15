import numpy as np

def get_class_weights(y):
    total = len(y)
    class_counts = np.bincount(y)
    weights = {i: total / count for i, count in enumerate(class_counts)}
    return weights