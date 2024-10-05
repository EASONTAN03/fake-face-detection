import numpy as np

def resize_and_normalize(image, size):
    image_resized = image.resize(size)
    image_array = np.array(image_resized) / 255.0  # Normalize to [0, 1]
    return image_array