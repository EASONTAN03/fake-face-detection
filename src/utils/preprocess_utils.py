import numpy as np
import cv2
from PIL import Image
from skimage.feature import local_binary_pattern

def resize_and_normalize(image_array, size, normalize=True):
    """
    Resize and normalize an image to the specified size.

    Args:
        image_array (np.ndarray): The input image as a NumPy array.
        size (tuple): The target size as (width, height).
        normalize (bool): Whether to normalize the image values to the range [0, 1].
        color_mode (str): The color mode to convert to ('RGB', 'YCbCr', or 'Gray').

    Returns:
        tuple: Original image array and resized, normalized image as NumPy arrays.
    """
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(image_array, size, interpolation=cv2.INTER_LANCZOS4)
    
    # Normalize the image if required
    if normalize:
        image_normalized = image_resized / 255.0
    else:
        image_normalized = image_resized
    
    return image_array, image_normalized

def denormalize_img(img):
    if img.dtype != np.uint8:
            img = (img * 255/np.max(img)).astype(np.uint8)
            return img  # Scale and convert to uint8 if necessary

'''Apply texture extraction'''
def apply_fft(images):
    """
    Apply Fast Fourier Transform (FFT) to each image to analyze frequency components.
    
    Args:
        images (list of np.ndarray): List of images to process.
        
    Returns:
        list of np.ndarray: List of FFT-processed images.
    """
    processed_images = []
    for img in images:
        f = np.fft.fft2(img)

        # Shift the zero-frequency component to the center
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)  # Use log for better visibility

        processed_images.append(magnitude_spectrum)
    return processed_images


def apply_lbp(images, radius=1, n_points=8, method='default'):
    """
    Apply Local Binary Pattern (LBP) to detect texture inconsistencies in the images.
    
    Args:
        images (list of np.ndarray): List of images to process.
        radius (int): Radius of the LBP pattern.
        n_points (int): Number of points considered in the LBP pattern.
        
    Returns:
        list of np.ndarray: List of LBP-processed images.
    """
    processed_images = []

    for img in images:
        lbp = local_binary_pattern(img, n_points, radius, method='uniform')
        processed_images.append(lbp)
    return processed_images

'''Apply edge detection'''
def apply_sobel(images, kernel=3):
    """
    Applies Sobel edge detection to each image in the list.

    The Sobel operator is a discrete differentiation operator that computes
    the gradient of an image intensity function in the horizontal and vertical
    directions. The gradient magnitude is then computed as the Euclidean norm
    of the horizontal and vertical gradients.

    Args:
        images (list of np.ndarray): List of images to process.
        kernel (int): Size of the Sobel kernel. Defaults to 3.

    Returns:
        list of np.ndarray: List of Sobel edge-detected images.
    """
    processed_images = []

    for img in images:
        # Apply Sobel edge detection
        # Compute the gradient in the X direction
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)  
        # Compute the gradient in the Y direction
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)  
        # Compute the magnitude of the gradient (Euclidean norm)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        # Append the Sobel edge-detected image to the list
        processed_images.append(sobel_combined)

    return processed_images

    

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
    # Apply CLAHE to the grayscale image
        clahe_image = clahe.apply(img)
        processed_images.append(clahe_image)
    return processed_images

def extract_statistics(image):
    stats = {
        'Mean': round(np.mean(image), 5),
        'Std Dev': round(np.std(image), 5),
        'Max': round(np.max(image), 5),
        'Min': round(np.min(image), 5),
        'Median': round(np.median(image), 5)
    }
    return stats
