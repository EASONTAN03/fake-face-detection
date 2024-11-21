import os
import shutil
import random
import numpy as np
import json
import csv
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess_utils import denormalize_img


def dict2str(model_params):
    if model_params is None:
        return " " 
    formatted_params = [f"{key}: {value}" for key, value in model_params.items()]
    return ', '.join(formatted_params)

def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def check_delete_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

def load_img(real_and_fake_face_path, datatype):

    real_images=[]
    fake_images=[]
    try:
        for index, dir_name in enumerate(datatype):
            dir_path = os.path.join(real_and_fake_face_path, dir_name)
            print(f"Directory: {dir_name}")
            print(f"Full path: {dir_path}")

            if os.path.isdir(dir_path):  
                for root, _, filenames in os.walk(dir_path):
                    print(f"Reading files in {root}")  
                    for filename in filenames:
                        if filename.endswith(('.jpg', '.jpeg', '.png')):  
                            file_path = os.path.join(root, filename)
                            if index == 0:  # Assuming the first item in datatype is for real images
                                real_images.append(file_path)
                            elif index == 1:  # Assuming the second item is for fake images
                                fake_images.append(file_path)
            else:
                print(f"Directory {dir_path} does not exist.")
        return real_images,fake_images
    except FileNotFoundError:
        print(f"The directory {real_and_fake_face_path} does not exist.")

def select_and_extract_images(real_images, fake_images, split_data=10, random_state=42):
    # Set the random seed for reproducibility
    random.seed(random_state)

    # Select random images
    selected_real = random.sample(real_images, min(split_data, len(real_images)))
    selected_fake = random.sample(fake_images, min(split_data, len(fake_images)))
    print(f"Extracted {len(selected_real)} real and {len(selected_fake)} fake images")
    return selected_real, selected_fake

def split_dataset(real_images, fake_images, split_ratio, random_state):
    # Combine real and fake images
    print("Splitting dataset......")
    images = real_images + fake_images
    labels = [0] * len(real_images) + [1] * len(fake_images)
    # Split the combined dataset into train and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(
        images,
        labels,
        test_size =split_ratio,
        random_state=random_state,
        stratify=labels

    )

    # Count the number of real and fake images in the train and test sets
    train_real_count = len([label for label in train_labels if label == 0])
    train_fake_count = len(train_labels) - train_real_count
    test_real_count = len([label for label in test_labels if label == 0])
    test_fake_count = len(test_labels) - test_real_count

    split_details={
    "Total train data": [train_real_count, train_fake_count],
    "Total test data": [test_real_count, test_fake_count],
    "Remark": "real, fake"
    }
    
    print("Done splitting dataset")
    return train_images, test_images, train_labels, test_labels, split_details

def write_csv(stats_dict, dir, file_name):
    # Check if the file exists and has the correct column names
    file_path = os.path.join(dir, file_name)
    if not os.path.exists(file_path):
        # Create the file and write the column names
        create_dir(file_path)
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = [
                'Benchmark', 'Dataset', 'Preprocess', 'Model', 'Model_name', 
                'tp', 'fp', 'tn', 'fn', 'accuracy', 'precision', 'recall', 
                'specificity', 'f1-score', 'mcc', 'auroc'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    # Append a new row to the CSV file
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = [
            'Benchmark', 'Dataset', 'Preprocess', 'Model', 'Model_name', 
            'tp', 'fp', 'tn', 'fn', 'accuracy', 'precision', 'recall', 
            'specificity', 'f1-score', 'mcc', 'auroc'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(stats_dict)

def write_json(dict, json_file_path):
    # Load existing logs if the file exists
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as f:
            old_data = json.load(f)
    else:
        old_data = []  # Initialize as an empty list if the file does not exist

    # Append new log data
    old_data.append(dict)  # Replace 'new_log_entry' with your actual log entry

    # Write updated logs back to the file
    with open(json_file_path, "w") as f:
        json.dump(old_data, f, indent=4)

    print("Updated log data")

def compute_stats(tn, tp, fp, fn):
    """
    Compute various measurement metrics from the given confusion matrix.

    Returns
        A dictionary containing the computed metrics.
    """
    # Compute the accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Compute the precision
    precision = tp / (tp + fp) if tp + fp > 0 else 0

    # Compute the recall
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    # Compute the specificity
    specificity = tn / (tn + fp) if tn + fp > 0 else 0

    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Compute the Matthews correlation coefficient (MCC)
    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = mcc_numerator / mcc_denominator if mcc_denominator != 0 else 0

    # Compute the AUROC (approximation)
    # Note: This is a simplified version and may not be as accurate as the library implementation
    auroc = 0.5 + ((tp - fn) * (tp + fn)) / (2 * (tp + fn) * (tn + fp))

    # Return the computed metrics
    stats_dir= {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        'f1-score': f1_score,
        "mcc": mcc,
        "auroc": auroc
    }

    return {key: round(value, 3) for key, value in stats_dir.items()}


def load_labels(label_file):
    return np.load(label_file)


def read_images(image_dir):
    """
    Read images from JPG files in the specified directory.

    Args:
        image_dir (str): Directory containing JPG files.

    Returns:
        List of images loaded from JPG files.
    """
    images = []
    
    # Loop through all files in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):  # Check for JPG files
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)  # Read the image using cv2
            
            if img is None:
                print(f"Error loading image: {img_path}")
                continue
                
            images.append(img)  # Append the loaded image to the list
    return images


def plot_images(original_images, processed_images, feature_images, labels, num_images=6):
    """
    Plot the first `num_images` original images in the first row, their processed versions in the second row,
    and extracted feature images in the third row, along with labels indicating whether the image is tampered or original.
    
    Args:
        original_images (list): List of original images.
        processed_images (list): List of processed images.
        feature_images (list): List of extracted feature images.
        labels (np.ndarray): Labels indicating tampered (1) or original (0) images.
        num_images (int): Number of images to plot. Default is 6.
    """
    plt.figure(figsize=(15, 9))  # Adjust height for three rows

    # Plot original images
    for i in range(num_images):
        plt.subplot(3, num_images, i + 1)
        plt.imshow(cv2.cvtColor(original_images[i], cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
        
        # Set title based on label
        label = "Fake" if labels[i] == 1 else "Real"
        plt.title(f'Original {i + 1} ({label})')
        plt.axis('off')
    
    # Plot processed images
    for i in range(num_images):
        plt.subplot(3, num_images, i + 1 + num_images)
        # Handle grayscale vs RGB images
        # if np.max(processed_images[i]) <= 1:
        #     processed_images_i = ((processed_images[i] * 255)).astype(np.uint8)
        # else:
        #     processed_images_i=processed_images[i].astype(np.uint8)
        processed_images_i=processed_images[i]
        if len(processed_images_i.shape) == 2:  # Grayscale image
            plt.imshow(processed_images_i, cmap='gray')  # Use gray colormap
            label = "Fake" if labels[i] == 1 else "Real"
            plt.title(f'Processed {i + 1} ({label})')
        else:  # RGB image
            if np.max(processed_images_i) <= 8:
    
                processed_images_i = denormalize_img(processed_images_i).astype(np.uint8)
            else:
                processed_images_i=processed_images_i.astype(np.uint8)
            plt.imshow(cv2.cvtColor(processed_images_i, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
            label = "Fake" if labels[i] == 1 else "Real"
            plt.title(f'Processed {i + 1} ({label})')

        plt.axis('off')
    
    # Plot feature images
    for i in range(num_images):
        plt.subplot(3, num_images, i + 1 + 2 * num_images)
        # if np.max(feature_images[i]) <= 1:
        #     feature_images_i = ((feature_images[i] * 255)).astype(np.uint8)
        # else:
        #     feature_images_i=feature_images[i].astype(np.uint8)
        feature_images_i=feature_images[i]
        if len(feature_images_i.shape) == 2:  # Grayscale feature image
            plt.imshow(feature_images_i, cmap='gray')  # Use gray colormap
            plt.title(f'Feature {i + 1}')
        else:  # RGB feature image
            if np.max(feature_images_i) <= 1:
                feature_images_i = denormalize_img(feature_images_i).astype(np.uint8)
            else:
                feature_images_i=feature_images_i.astype(np.uint8)
            plt.imshow(cv2.cvtColor(feature_images_i, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
            plt.title(f'Feature {i + 1}')
        plt.axis('off')
    
    # Improve layout and spacing
    plt.tight_layout()
    plt.suptitle('Comparison of Original, Processed, and Feature Images', fontsize=16)  # Add a title for the entire plot
    plt.show()
