import numpy as np


def compress_image(image, k):
    """
    Compress a grayscale image using Singular Value Decomposition (SVD).

    Parameters:
        image : numpy.ndarray
            Grayscale image represented as a 2D matrix.
        k : int
            Number of singular values to retain.

    Returns:
        numpy.ndarray
            Reconstructed image using the first k singular values.
    """

    if image.ndim != 2:
        raise ValueError("Input image must be a grayscale 2D matrix.")

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    # Perform SVD
    U, S, VT = np.linalg.svd(image, full_matrices=False)

    # Make sure k does not exceed the number of available singular values
    k = min(k, len(S))

    # Keep only the first k components
    U_k = U[:, :k]
    S_k = S[:k]
    VT_k = VT[:k, :]

    # Reconstruct the image
    reconstructed = U_k @ np.diag(S_k) @ VT_k

    return reconstructed