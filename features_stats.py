import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

processed_data_path = config['dataset']['processed']
dataset = config['configs']['dataset']
benchmark = config['configs']['benchmark']

with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)
prepare_benchmark = params['prepare']['benchmark']
train_test = params['prepare']['train_test']

data_dir = f'{dataset}_{benchmark}'
input_dir = os.path.join(processed_data_path, data_dir, train_test)

# Path to the .npy files
features_path = os.path.join(input_dir,f'features_{prepare_benchmark}.npy')
labels_path = os.path.join(input_dir,f'labels_{prepare_benchmark}.npy')

# Load the features and labels
features = np.load(features_path)  # Shape: (N, 224, 224, 3)
labels = np.load(labels_path)  # Shape: (N,)

# Ensure labels are 0 (fake) and 1 (real)
real_images = features[labels == 1]  # Real face images
fake_images = features[labels == 0]  # Fake face images

# Helper function to compute and print statistics
def compute_stats(images, label_name):
    images = images.astype(np.float32)  # Ensure the images are in float format
    flat_images = images.ravel()  # Flatten the image array to 1D for easier stats calculation
    
    # Compute statistics
    mean = np.mean(flat_images)
    std = np.std(flat_images)
    var = np.var(flat_images)
    min_val = np.min(flat_images)
    q1 = np.percentile(flat_images, 25)
    median = np.percentile(flat_images, 50)  # or np.median(flat_images)
    q3 = np.percentile(flat_images, 75)
    max_val = np.max(flat_images)
    
    # Print statistics
    print(f"Statistics for {label_name} images:")
    print(f"Mean: {mean:.4f}")
    print(f"Standard Deviation: {std:.4f}")
    print(f"Variance: {var:.4f}")
    print(f"Min: {min_val:.4f}")
    print(f"Q1 (25th percentile): {q1:.4f}")
    print(f"Median (Q2/50th percentile): {median:.4f}")
    print(f"Q3 (75th percentile): {q3:.4f}")
    print(f"Max: {max_val:.4f}")
    print("-" * 40)
    
    return {
        'mean': mean, 
        'std': std, 
        'var': var, 
        'min': min_val, 
        'q1': q1, 
        'median': median, 
        'q3': q3, 
        'max': max_val
    }

# Function to plot histogram
def plot_histogram(images, label_name):
    plt.figure(figsize=(6, 4))
    plt.hist(images.ravel(), bins=256, color='blue', alpha=0.7)
    plt.title(f'Pixel Intensity Histogram: {label_name}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

# Compute statistics and plot histograms
print("Computing statistics for combined, real, and fake images...")

# Combined statistics
combined_stats = compute_stats(features, "Combined (Real + Fake)")
plot_histogram(features, "Combined (Real + Fake)")

# Real images statistics
real_stats = compute_stats(real_images, "Real")
plot_histogram(real_images, "Real")

# Fake images statistics
fake_stats = compute_stats(fake_images, "Fake")
plot_histogram(fake_images, "Fake")

# Store computed stats in a dictionary for later use (optional)
stats = {
    'combined': combined_stats,
    'real': real_stats,
    'fake': fake_stats
}
print("Statistics computation completed and histograms generated.")
