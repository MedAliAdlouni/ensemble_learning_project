import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def calculate_proportions(dataset, dataset_name):
    labels = dataset[:, 1]
    unique, counts = np.unique(labels, return_counts=True)
    proportions = {int(k): round(float(v), 4) for k, v in zip(unique, counts / len(labels))}
    print(f"{dataset_name}: {proportions}")


# DOMAIN ADAPTATION

# Function to subsample data from source and target datasets
def subsample_data(X_source, X_target, n_samples=1000):
    # Ensure that the size of the subset is smaller than the number of samples
    X_source_sub = X_source[np.random.choice(X_source.shape[0], n_samples, replace=False)]
    X_target_sub = X_target[np.random.choice(X_target.shape[0], n_samples, replace=False)]
    return X_source_sub, X_target_sub


# Function to calculate Maximum Mean Discrepancy (MMD) between two datasets
def calculate_mmd(X_source, X_target, kernel='rbf', gamma=1.0):
    # Apply kernel functions (Radial Basis Function Kernel here)
    if kernel == 'rbf':
        K_ss = rbf_kernel(X_source, X_source, gamma=gamma)
        K_tt = rbf_kernel(X_target, X_target, gamma=gamma)
        K_st = rbf_kernel(X_source, X_target, gamma=gamma)

    # Calculate MMD
    mmd = np.mean(K_ss) + np.mean(K_tt) - 2 * np.mean(K_st)
    return mmd
