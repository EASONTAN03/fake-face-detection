import numpy as np
from sklearn.decomposition import PCA
import cv2
from PIL import Image
import cv2
from skimage.feature import hog, local_binary_pattern
from skimage import exposure
import dlib
import io
import pywt



def resize_and_normalize(image_array, size, normalize=True, color_mode='RGB'):
    """
    Resize and normalize an image to the specified size.
    
    Args:
        image_array (np.ndarray): The input image as a NumPy array with shape (height, width, channels).
        size (tuple): The target size as (width, height).
        
    Returns:
        np.ndarray: Resized and normalized image as a NumPy array.
    """
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(image_array)

    # Apply color conversion based on the color_mode argument
    if color_mode == 'YCbCr':
        image = image.convert('YCbCr')
    elif color_mode == 'Gray':
        image = image.convert('L')  # 'L' mode for grayscale
    else:
        image = image.convert('RGB')  # Default to 'RGB' if no valid mode is passed
    
    # Resize the image
    image_resized = image.resize(size, Image.LANCZOS)  
    image_normalized = np.array(image_resized) / 255.0 if normalize else np.array(image_resized)   
    
    return image_array,image_normalized

def denormalize_img(img):
    if img.dtype != np.uint8:
            img = (img * 255/np.max(img)).astype(np.uint8)
            return img  # Scale and convert to uint8 if necessary

    
'''Apply edge detection and image filter techniques'''
def apply_sobel(images):
    """
    Apply Sobel edge detection to each image.
    
    Args:
        images (list of np.ndarray): List of images to process.
    
    Returns:
        list of np.ndarray: List of Sobel edge-detected images.
    """
    processed_images = []
    
    for img in images:
        # Apply Sobel edge detection
        
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # X direction
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Y direction
        sobel_combined = cv2.magnitude(sobelx, sobely)
        
        processed_images.append(sobel_combined)
    
    return processed_images


def apply_gaussian(images, kernel_size=(5, 5), sigma=1.0):
    """
    Apply Gaussian smoothing to each image.
    
    Args:
        images (list of np.ndarray): List of images to process.
        kernel_size (tuple): The kernel size for the Gaussian blur.
        sigma (float): The standard deviation for the Gaussian kernel.
    
    Returns:
        list of np.ndarray: List of blurred images.
    """
    processed_images = []
    
    for img in images:
        # Apply Gaussian blur
        blurred_img = cv2.GaussianBlur(img, kernel_size, sigma)
        processed_images.append(blurred_img)
    
    return processed_images

def apply_gabor(images, sigma=5, frequency=0.6):
    """
    Apply Gabor filter to detect texture inconsistencies in images.
    
    Args:
        images (list of np.ndarray): List of images to process.
        frequency (float): Frequency of the Gabor filter.
    
    Returns:
        list of np.ndarray: List of Gabor-filtered images.
    """
    processed_images = []

    for img in images:
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)  # Scale and convert to uint8 if necessary

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[-1] == 3 else img
        gabor_image = cv2.getGaborKernel((21, 21), sigma, np.pi/4, frequency, 0.5, 0, ktype=cv2.CV_32F)
        filtered_img = cv2.filter2D(gray_img, cv2.CV_8UC3, gabor_image)
        processed_images.append(filtered_img)
    
    return processed_images


'''Apply feacture extraction, enhancement and reduction techniques'''
def apply_pca(images, components=2, enhancement='stretch contrast'):
    """
    Apply PCA on a list of images and reduce the dimensions to the specified number of components.
    Optionally, apply contrast stretching enhancement.
    
    Args:
        images (list of np.ndarray): List of images to process.
        components (int): Number of PCA components to project down to.
        enhancement (str): Enhancement method. 'stretch contrast' is the current enhancement method.
        
    Returns:
        list of np.ndarray: List of PCA-transformed images.
    """
    processed_images = []
    flattened_images = np.array([img.flatten() for img in images])
    print(f"Flattened images shape: {flattened_images.shape}")
    
    # Apply PCA on the dataset of flattened images
    components_adjusted = min(components * 10, flattened_images.shape[1])
    pca = PCA(n_components=components_adjusted)
    pca_result = pca.fit_transform(flattened_images)
    
    # Inverse PCA to get back to image space but with reduced components
    pca_images = pca.inverse_transform(pca_result)
    
    # Reshape each processed image back to its original shape
    processed_images = [img.reshape(images[0].shape) for img in pca_images]
    
    # Apply contrast stretching if requested
    if enhancement == 'stretch contrast':
        processed_images = [stretch_contrast(img) for img in processed_images]
    
    return processed_images

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
        # Convert to grayscale if needed
        # if np.max(img)<=1:
        #     img = denormalize_img(img)
        # Apply FFT
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift)) 
        processed_images.append(magnitude_spectrum)
    return processed_images

def apply_sift(images):
    """
    Apply SIFT to each image and return the image with key points drawn.
    
    Args:
        images (list of np.ndarray): List of images to process.
    
    Returns:
        list of np.ndarray: List of images with key points drawn.
    """
    processed_images = []
    sift = cv2.SIFT_create()

    for img in images:
        # Detect SIFT key points and descriptors
        if img.dtype != np.uint8:
            img_i = (img * 255).clip(0, 255).astype(np.uint8)  # Ensure the image is within range and convert
        keypoints, descriptors = sift.detectAndCompute(img_i, None)
        # Draw key points on the image
        img_with_keypoints = cv2.drawKeypoints(img_i, keypoints, None)
        processed_images.append(img_with_keypoints)
    return processed_images


def apply_hog(images):
    """
    Apply HOG to each image and return the HOG descriptor visualization.
    
    Args:
        images (list of np.ndarray): List of images to process.
    
    Returns:
        list of np.ndarray: List of HOG descriptor images.
    """
    processed_images = []
    for img in images:        
        # Compute HOG features and visualize them
        hog_features, hog_image = hog(img, pixels_per_cell=(16, 16),
                                      cells_per_block=(2, 2), visualize=True)
        # Rescale the HOG image for better visualization
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        processed_images.append(hog_image_rescaled)
    return processed_images

def apply_ela(images, quality=90):
    """
    Apply Error Level Analysis (ELA) to detect image manipulation.
    
    Args:
        images (list of np.ndarray): List of images to process.
        quality (int): Quality level for saving the image for ELA.
        
    Returns:
        list of np.ndarray: List of ELA-processed images.
    """
    processed_images = []

    for img in images:
        # Convert the NumPy array to a PIL image
        pil_img = Image.fromarray(img)
         # Check and convert to RGB mode if needed
        if pil_img.mode == 'F':  # If it's in floating point grayscale
            pil_img = pil_img.convert('L')  # Convert to 8-bit grayscale (L) or use 'RGB' for color

        elif pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')  # Convert to RGB if it's not in that mode
        # Save the image with specified quality
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        # Reload the image and compute the difference
        pil_img_reloaded = Image.open(buffer)
        ela_image = np.abs(np.array(pil_img, dtype=np.float32) - np.array(pil_img_reloaded, dtype=np.float32))
        # Normalize the result
        ela_image = (ela_image / ela_image.max()) * 255.0
        processed_images.append(ela_image.astype(np.uint8))
    
    return processed_images


def apply_lbp(images, radius=1, n_points=8):
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


def apply_landmark_detection(images, predictor_path='shape_predictor_68_face_landmarks.dat'):
    """
    Apply facial landmark detection to detect inconsistencies in face structure.
    
    Args:
        images (list of np.ndarray): List of images to process.
        predictor_path (str): Path to the pre-trained facial landmark model.
        
    Returns:
        list of np.ndarray: List of images with facial landmarks drawn.
    """
    processed_images = []
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    for img in images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = detector(gray_img)
        
        for face in faces:
            landmarks = predictor(gray_img, face)
            
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
        
        processed_images.append(img)
    
    return processed_images

def apply_dct(images):
    """
    Apply Discrete Cosine Transform (DCT) to each image.
    
    Args:
        images (list of np.ndarray): List of images to process.
        
    Returns:
        list of np.ndarray: List of DCT-processed images.
    """
    processed_images = []

    for img in images:
        # Convert to grayscale if needed
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[-1] == 3 else img
        
        # Apply DCT
        dct_image = cv2.dct(np.float32(gray_img))
        
        # Scale the DCT coefficients for better visualization
        dct_image_scaled = cv2.normalize(dct_image, None, 0, 255, cv2.NORM_MINMAX)
        processed_images.append(dct_image_scaled.astype(np.uint8))
    
    return processed_images


def apply_wavelet(images, wavelet='haar'):
    """
    Apply Wavelet Transform to each image.
    
    Args:
        images (list of np.ndarray): List of images to process.
        wavelet (str): Name of the wavelet to use for transformation.
        
    Returns:
        list of np.ndarray: List of wavelet-processed images.
    """
    processed_images = []

    for img in images:
        # Convert to grayscale if needed
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[-1] == 3 else img
        
        # Apply 2D Discrete Wavelet Transform
        coeffs2 = pywt.dwt2(gray_img, wavelet)
        cA, (cH, cV, cD) = coeffs2  # Approximation and detail coefficients
        
        # Combine coefficients for visualization (you can choose how to visualize)
        combined_image = np.hstack((cA, cH, cV, cD))
        processed_images.append(combined_image)
    
    return processed_images


def stretch_contrast(image):
    """
    Apply contrast stretching to an image using min-max normalization.
    
    Args:
        image (np.ndarray): Input image.
        
    Returns:
        np.ndarray: Contrast-stretched image.
    """
    # Normalize the pixel values between 0 and 255
    min_val, max_val = np.min(image), np.max(image)
    stretched_img = 255 * (image - min_val) / (max_val - min_val)
    
    return stretched_img.astype(np.uint8)
