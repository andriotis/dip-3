import matplotlib.pyplot as plt
import numpy as np
import cv2
import hw3_helper_utils
from scipy.ndimage import convolve


def show_images_grid(images, titles=None):
    num_images = len(images)
    if titles is None:
        titles = [""] * num_images

    # Calculate the number of rows needed for 2 columns
    num_rows = (num_images + 1) // 2 if num_images > 1 else 1

    fig, axes = plt.subplots(num_rows, 2, figsize=(5, num_rows * 5))

    # Flatten axes for easier iteration if more than 1 image
    if num_images > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Wrap single axis in a list

    for idx, (img, title) in enumerate(zip(images, titles)):
        ax = axes[idx]
        M, N = img.shape
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    # Hide any unused subplots if number of images is odd
    for idx in range(num_images, num_rows * 2):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


def generate_noisy_image(image_path, noise_level=0.02, length=20, angle=30):
    x = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    x = (x - np.min(x)) / (np.max(x) - np.min(x))

    # create white noise with noise level
    v = noise_level * np.random.randn(*x.shape)

    # create motion blur filter
    h = hw3_helper_utils.create_motion_blur_filter(length=length, angle=angle)

    # obtain the filtered image
    y0 = convolve(x, h, mode="wrap")

    # generate the noisy image
    y = y0 + v

    return x, y0, y, h, v


def pad_array(array, shape):
    padded_array = np.zeros(shape)
    start_row = (shape[0] - array.shape[0]) // 2
    start_col = (shape[1] - array.shape[1]) // 2
    padded_array[
        start_row : start_row + array.shape[0], start_col : start_col + array.shape[1]
    ] = array
    return padded_array
