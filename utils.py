import numpy as np
from utils import *
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
import numpy as np
import cv2
import hw3_helper_utils
from scipy.ndimage import convolve
import sys


def calculate_mse(x: np.ndarray, x_hat: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) between two images.

    Args:
    x (np.ndarray): Original image.
    x_hat (np.ndarray): Estimated or reconstructed image.

    Returns:
    float: The Mean Squared Error between x and x_hat.
    """
    return float(np.mean((x - x_hat) ** 2))


def optimize_k(
    x: np.ndarray, y: np.ndarray, h: np.ndarray, k_range: np.ndarray
) -> tuple:
    """
    Optimize the K parameter for Wiener filtering by minimizing MSE.

    Args:
    x (np.ndarray): Original image.
    y (np.ndarray): Blurred and noisy image.
    h (np.ndarray): Motion blur filter.
    k_range (np.ndarray): Range of K values to try.

    Returns:
    tuple:
        - best_k (float): The optimal K value.
        - best_mse (float): The MSE corresponding to the best K.
        - mse_values (list): List of MSE values for each K in k_range.
    """
    best_k = None
    best_mse = float("inf")
    mse_values = []

    for k in k_range:
        # Apply Wiener filter with current K
        x_hat = my_wiener_filter(y, h, k)

        # Calculate MSE
        mse = calculate_mse(x, x_hat)
        mse_values.append(mse)

        # Update best K if current MSE is lower
        if mse < best_mse:
            best_mse = mse
            best_k = k

    return best_k, best_mse, mse_values


def generate_noisy_image(
    image_path: str, noise_level: float = 0.02, length: int = 20, angle: int = 30
) -> tuple:
    """
    Generate a blurred and noisy version of an input image.

    Args:
    image_path (str): Path to the input image file.
    noise_level (float): Standard deviation of the Gaussian noise (default: 0.02).
    length (int): Length of the motion blur filter (default: 20).
    angle (int): Angle of the motion blur in degrees (default: 30).

    Returns:
    tuple:
        - x (np.ndarray): Original image, normalized to [0, 1].
        - y0 (np.ndarray): Blurred image without noise.
        - y (np.ndarray): Blurred and noisy image.
        - h (np.ndarray): Motion blur filter.
        - v (np.ndarray): Noise array.
    """
    # Load and normalize the image
    x = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    x = (x - np.min(x)) / (np.max(x) - np.min(x))

    # Generate Gaussian noise
    v = noise_level * np.random.randn(*x.shape)

    # Create motion blur filter
    h = hw3_helper_utils.create_motion_blur_filter(length=length, angle=angle)

    # Apply motion blur
    y0 = convolve(x, h, mode="wrap")

    # Add noise to the blurred image
    y = y0 + v

    return x, y0, y, h, v


def my_wiener_filter(y: np.ndarray, h: np.ndarray, K: float) -> np.ndarray:
    """
    Apply Wiener filter for image deblurring.

    Args:
    y (np.ndarray): Blurred and noisy input image.
    h (np.ndarray): Motion blur filter.
    K (float): Regularization parameter.

    Returns:
    np.ndarray: Deblurred image.
    """
    # Get the dimensions of the input image
    M, N = y.shape
    L, P = h.shape

    # Pad h to match the size of y
    h_padded = np.zeros((M, N))
    h_padded[:L, :P] = h
    h_padded = np.roll(h_padded, (-L // 2, -P // 2), axis=(0, 1))

    # Compute DFT of the image and the blur kernel
    Y = fft2(y)
    H = fft2(h_padded)

    # Compute power spectrum of the blur kernel
    H_mag_sq = np.abs(H) ** 2

    # Apply Wiener filter
    G = np.conj(H) / (H_mag_sq + 1 / K)
    X_hat = Y * G

    # Inverse DFT to get the deblurred image
    x_hat = np.real(ifft2(X_hat))

    # Ensure the result is in the original range [0, 1]
    x_hat = np.clip(x_hat, 0, 1)

    return x_hat


def plot_wiener_filter_results(
    x, y, h, k_range, best_k, best_mse, mse_values, x_hat_optimal, verbose=False
):
    """
    Plot the results of Wiener filter optimization and image deblurring.

    Args:
    x (np.ndarray): Original image.
    y (np.ndarray): Blurred and noisy image.
    h (np.ndarray): Motion blur filter.
    k_range (np.ndarray): Range of K values used for optimization.
    best_k (float): Optimal K value.
    best_mse (float): MSE corresponding to the optimal K.
    mse_values (list): List of MSE values for each K.
    x_hat_optimal (np.ndarray): Optimally deblurred image.
    verbose (bool): If True, plot all figures together. If False, plot independently. Default is False.
    """
    if verbose:
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle("Wiener Filter Optimization and Results", fontsize=16)

    # Plot 1: Original MSE curve
    if verbose:
        ax1 = fig.add_subplot(231)
    else:
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
    ax1.semilogx(k_range, mse_values)
    ax1.set_xlabel("K value")
    ax1.set_ylabel("Mean Squared Error")
    ax1.set_title("MSE vs K")
    ax1.grid(True)
    if not verbose:
        plt.show()

    # Plot 2: MSE curve with optimal K
    if verbose:
        ax2 = fig.add_subplot(232)
    else:
        plt.figure(figsize=(10, 6))
        ax2 = plt.gca()
    ax2.semilogx(k_range, mse_values, "b-", label="MSE curve")
    ax2.semilogx(best_k, best_mse, "ro", markersize=10, label="Optimal K")
    ax2.annotate(
        f"Optimal: (K={best_k:.4f}, MSE={best_mse:.4e})",
        xy=(best_k, best_mse),
        xytext=(best_k * 1.5, best_mse * 1.1),
        arrowprops=dict(facecolor="black", shrink=0.05),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1),
    )
    ax2.set_xlabel("K value")
    ax2.set_ylabel("Mean Squared Error")
    ax2.set_title("MSE vs K (with Optimal K)")
    ax2.grid(True)
    ax2.legend()
    ax2.set_ylim(0, min(max(mse_values), best_mse * 2))
    if not verbose:
        plt.show()

    # Plot 3, 4, 5: Original, Blurred/Noisy, and Deblurred Images
    if verbose:
        ax3 = fig.add_subplot(234)
        ax4 = fig.add_subplot(235)
        ax5 = fig.add_subplot(236)
    else:
        fig, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(15, 5))

    ax3.imshow(x, cmap="gray")
    ax3.set_title("Original Image")
    ax3.axis("off")

    ax4.imshow(y, cmap="gray")
    ax4.set_title("Blurred and Noisy Image")
    ax4.axis("off")

    ax5.imshow(x_hat_optimal, cmap="gray")
    ax5.set_title(f"Deblurred Image (K={best_k:.4f})")
    ax5.axis("off")

    if verbose:
        plt.tight_layout()
    plt.show()


# Example usage
def main():

    # Load and preprocess images
    user_params = {
        "image_path": sys.argv[1],
        "noise_level": float(sys.argv[2]),
        "length": int(sys.argv[3]),
        "angle": int(sys.argv[4]),
    }

    x, y0, y, h, v = generate_noisy_image(**user_params)

    # Define a range of K values to try
    k_range = np.logspace(
        -3, 3, 100
    )  # 100 points logarithmically spaced from 10^-3 to 10^3

    # Optimize K
    best_k, best_mse, mse_values = optimize_k(x, y, h, k_range)
    print(f"Best K: {best_k}")
    print(f"Best MSE: {best_mse}")

    # Get optimal deblurred image
    x_hat_optimal = my_wiener_filter(y, h, best_k)

    # Plot results
    plot_wiener_filter_results(
        x, y, h, k_range, best_k, best_mse, mse_values, x_hat_optimal, verbose=True
    )


if __name__ == "__main__":
    main()
