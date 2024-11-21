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

def compute_histogram(image, bins=9, range=(0,256)):
    """
    Compute a histogram of the power spectrum for the given image.

    Args:
        image (np.ndarray): Image to compute histogram on.
        bins (int): Number of histogram bins.
        range (tuple): Range for histogram bins.
        
    Returns:
        np.ndarray: Normalized histogram of the image.
    """
    histogram, bin_edges = np.histogram(image, bins=bins, range=range)
    histogram = histogram / np.sum(histogram)  # Normalize histogram
    # print(bin_edges)
    # print(histogram)
    # histogram = histogram / np.sum(histogram)  # Normalize histogram
    # print(histogram)
    return histogram

'''Apply texture extraction'''
def apply_fft(images, bins, compute_hist):
    """
    Apply Fast Fourier Transform (FFT) to each image to analyze frequency components.
    
    Args:
        images (list of np.ndarray): List of images to process.
        
    Returns:
        list of np.ndarray: List of FFT-processed images.
    """
    channel_histograms = []
    print(images.shape)
    if len(images.shape) == 4:
        for channel in range(3):
            # Compute FFT for each channel separately
            f = np.fft.fft2(images[:, :, :, channel])
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = np.log(np.abs(fshift) + 1)
            
            # Optionally compute histogram
            if compute_hist:
                global_min = np.min(magnitude_spectrum)
                global_max = np.max(magnitude_spectrum)
                hist_range = (global_min, global_max)
                hist=[compute_histogram(img, bins, range=hist_range) for img in magnitude_spectrum]
                channel_histograms.append(hist)
            else:
                channel_histograms.append(magnitude_spectrum)
        channel_histograms = np.stack(channel_histograms, axis=-1)  # Shape: (409, 18, 3)
    else:
    # Compute FFT for all images at once
        f = np.fft.fft2(np.array(images), axes=(-2, -1))
        # Shift the zero-frequency component to the center
        fshift = np.fft.fftshift(f, axes=(-2, -1))

        # Compute the magnitude spectrum using log scale for better visibility
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        if compute_hist:
            global_min = np.min(magnitude_spectrum)
            global_max = np.max(magnitude_spectrum)
            hist_range = (global_min, global_max)
            hist=[compute_histogram(img, bins, range=hist_range) for img in magnitude_spectrum]
            channel_histograms=hist
        else:
            channel_histograms=magnitude_spectrum
    return channel_histograms

def apply_lbp(images, radius=1, n_points=8, method='default', compute_hist=False):
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
        # Check if the image is 3-channel (RGB)
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Process each channel independently for RGB images
            lbp_channels = []
            for channel in range(3):  # Iterate over the 3 channels (R, G, B)
                channel_lbp = local_binary_pattern(img[:, :, channel], n_points, radius, method)
                if compute_hist and method=='default':
                    lbp_hist, _ = np.histogram(channel_lbp, bins=range(0, 256), density=True)
                    lbp_channels.append(lbp_hist)
                elif compute_hist and not method=='default':
                    lbp_hist, _ = np.histogram(channel_lbp, bins=range(0, n_points + 3), density=True)
                    lbp_channels.append(lbp_hist)
                else:
                    lbp_channels.append(channel_lbp)
            # Stack the histograms or LBP outputs across the 3 channels
            processed_images.append(np.stack(lbp_channels, axis=-1))  # Shape: (height, width, 3)

        # Process a single-channel grayscale image
        elif len(img.shape) == 2:
            lbp = local_binary_pattern(img, n_points, radius, method)
            if compute_hist and method=='default':
                lbp_hist, _ = np.histogram(lbp, bins=range(0, 256), density=True)
                processed_images.append(lbp_hist)
            elif compute_hist and not method=='default':
                lbp_hist, _ = np.histogram(lbp, bins=range(0, n_points + 3), density=True)
            else:
                processed_images.append(lbp)
    
    return processed_images

'''Apply edge detection'''
def apply_sobel(images, kernel=3, bins=9, compute_hist=True):
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
        sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
        if compute_hist==True:
            range=(0,255)
            sobel_combined=compute_histogram(sobel_combined, bins=bins, range=range)
        processed_images.append(sobel_combined)

    return processed_images

    
def apply_clahe(images, clip_limit=2.0, tile_grid_size=(8, 8), bins=9, compute_hist=False):
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
        if len(images.shape) == 4:
            channel_histograms = []
            for channel in range(3):
                # Compute FFT for each channel separately
                clahe_image = clahe.apply(img[:, :, channel])

                # Optionally compute histogram
                if compute_hist:
                    global_min = np.min(clahe_image)
                    global_max = np.max(clahe_image)
                    hist_range = (global_min, global_max)
                    hist=compute_histogram(clahe_image, bins=bins, range=hist_range)
                    channel_histograms.append(hist)
                else:
                    channel_histograms.append(clahe_image)
            channel_histograms = np.stack(channel_histograms, axis=-1)  # Shape: (409, 18, 3)
            processed_images.append(channel_histograms)
        else:
            clahe_image = clahe.apply(img)
            if compute_hist:
                global_min = np.min(clahe_image)
                global_max = np.max(clahe_image)
                hist_range = (global_min, global_max)
                hist=compute_histogram(clahe_image, bins=bins, range=hist_range)
                processed_images.append(hist)
            else:
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

    # Convert the dictionary values to a list
    stats_list = list(stats.values())
    
    # Convert the list to a NumPy array
    stats_array = np.array(stats_list)

    return stats_array

# def extract_statistics(gray_image):
#     # Define patch size
#     patch_size = 32

#     # Step 1: Reshape the image into a 3D array of 7x7 patches, each of size 32x32
#     # (224x224) -> (7, 32, 7, 32), then rearrange to (7x7, 32, 32) for each patch
#     patches = gray_image.reshape(7, patch_size, 7, patch_size).swapaxes(1, 2).reshape(-1, patch_size, patch_size)

#     # Step 2: Calculate descriptive statistics for each patch in a vectorized way
#     min_vals = patches.min(axis=(1, 2))
#     q1_vals = np.percentile(patches, 25, axis=(1, 2))
#     q2_vals = np.median(patches, axis=(1, 2))
#     q3_vals = np.percentile(patches, 75, axis=(1, 2))
#     max_vals = patches.max(axis=(1, 2))
#     std_vals = patches.std(axis=(1, 2))
#     mean_vals = patches.mean(axis=(1, 2))

#     # Step 3: Stack all statistics to form the final feature matrix (49 patches x 7 features)
#     final_features = np.stack([min_vals, q1_vals, q2_vals, q3_vals, max_vals, std_vals, mean_vals], axis=1)

#     return final_features

def extract_with_mtcnn(image):
    from mtcnn import MTCNN
    # Initialize MTCNN detector
    detector = MTCNN()
    detections = detector.detect_faces(image)
    # Initialize a 180x180 grid (5x5 of 36x36 pixels each)
    concatenated_image = np.zeros((180, 180, 3), dtype=np.uint8)
    
    for i, det in enumerate(detections[:1]):  # Limit to first detected face
        keypoints = det['keypoints']
        for j, (point_name, point) in enumerate(keypoints.items()):
            x, y = point
            # Crop 36x36 region around the keypoint
            cropped_region = image[max(y-18, 0):y+18, max(x-18, 0):x+18]
            resized_region = cv2.resize(cropped_region, (36, 36))
            
            # Place each 36x36 region into the 5x5 grid
            row = j // 5 * 36
            col = j % 5 * 36
            concatenated_image[row:row+36, col:col+36] = resized_region

    return concatenated_image