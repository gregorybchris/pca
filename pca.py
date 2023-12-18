import numpy as np


def pca(data: np.ndarray, n_components: int) -> np.ndarray:
    # Standardize the data (mean centering)
    mean = np.mean(data, axis=0)
    std = data - mean

    # Calculate the covariance matrix
    cov_matrix = np.cov(std, rowvar=False)

    # Compute the eigenvectors and eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvectors based on eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top 'n_components' eigenvectors
    principal_components = eigenvectors[:, :n_components]

    # Project the original data onto the principal components
    reduced_data = np.dot(std, principal_components)

    return reduced_data
