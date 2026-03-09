from jax import numpy as jnp

def cov_intrinsic(C, dim_eff, eigenvalue_threshold=1e-8):
    # Eigendecomposition
    eigenvalues, eigenvectors = jnp.linalg.eigh(C)
    # Sort eigenvalues in descending order
    idx = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Create mask for first m components
    n_features = eigenvectors.shape[1]
    mask = (jnp.arange(n_features) < dim_eff) & (eigenvalues > eigenvalue_threshold)

    # Apply mask to actually truncate (not zero out)
    V_full = eigenvectors[:, mask]
    Lambda_full = eigenvalues[mask]

    # Compute inverse eigenvalues (epsilon just for numerical safety)
    Lambda_inv = 1.0 / (Lambda_full + 1e-12)

    return Lambda_inv, V_full

def mahal_squared_eff(b, Lambda_inv, V_full):
    """
    Compute Mahalanobis distance between point b and samples in effective subspace E of dimension dim_eff.

    Parameters:
    -----------
    b : array-like, query point in ambient space, centered at mean (b = log(mean, b))
    Lambda_inv : inv of covariance matrix on E
    V_full : projection to effective subspace E, where data live
    eigenvalue_threshold : float, relative threshold for considering eigenvalues as non-zero

    Returns:
    --------
    distance : float, Mahalanobis squared distance from b to samples
    """
    # Project to effective subspace E
    b_proj = V_full.T @ b

    # Compute Mahalanobis distance in subspace
    # Only the first m components will contribute (others are multiplied by 0)
    mahal_squared = jnp.sum(b_proj * Lambda_inv * b_proj)

    return mahal_squared