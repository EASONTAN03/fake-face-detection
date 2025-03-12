import numpy as np
import cv2
from PIL import Image
from skimage.feature import local_binary_pattern

def apply_clahe(images, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    Parameters:
    - image: Input image (grayscale or color).
    - clip_limit: Threshold for contrast limiting.
    - tile_grid_size: Size of grid for histogram equalization (height, width).

    Returns:
    - clahe_image: Image after applying CLAHE.
    """
    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    processed_images = []

    for img in images:
        if len(images.shape) == 4:
            channel_histograms = []
            for channel in range(3):
                # Compute FFT for each channel separately
                clahe_image = clahe.apply(img[:, :, channel])
                channel_histograms.append(clahe_image)
            channel_histograms = np.stack(channel_histograms, axis=-1)  # Shape: (409, 18, 3)
            processed_images.append(channel_histograms)
        else:
            # Apply CLAHE to the grayscale image
            clahe_image = clahe.apply(img)
            processed_images.append(clahe_image)
    return processed_images
