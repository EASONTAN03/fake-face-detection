import os
import shutil
import random
import numpy as np
import json
import csv
import os
from sklearn.model_selection import train_test_split

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

    # Split the combined dataset into train and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(
        images,
        [0] * len(real_images) + [1] * len(fake_images),
        test_size =split_ratio,
        random_state=random_state
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
                'specificity', 'mcc', 'auroc'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    # Append a new row to the CSV file
    with open(file_path, 'a', newline='') as csvfile:
        # fieldnames = [
        #     'Benchmark', 'Dataset', 'Preprocess', 'Model', 'Model_name', 
        #     'tp', 'fp', 'tn', 'fn', 'accuracy', 'precision', 'recall', 
        #     'specificity', 'mcc', 'auroc'
        # ]
        # writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
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
        "mcc": mcc,
        "auroc": auroc
    }

    return round(stats_dir, 3)

def load_labels(label_file):
    return np.load(label_file)