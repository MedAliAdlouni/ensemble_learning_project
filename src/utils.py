import numpy as np

def calculate_proportions(dataset, dataset_name):
    labels = dataset[:, 1]
    unique, counts = np.unique(labels, return_counts=True)
    proportions = {int(k): round(float(v), 4) for k, v in zip(unique, counts / len(labels))}
    print(f"{dataset_name}: {proportions}")
