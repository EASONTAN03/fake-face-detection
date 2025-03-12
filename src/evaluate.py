import yaml
import csv
import os 
import cv2
import numpy as np
import pandas as pd
import json
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import joblib

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import operation_utils as utils
from modeling_utils import *

base_path = os.path.abspath(os.path.join(os.getcwd(), '..'))

with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

processed_data_path = os.path.join(base_path,config['dataset']['processed'])
output_model_path = os.path.join(base_path,config['output']['models'])

with open('../params.yaml', 'r') as file:
    params = yaml.safe_load(file)
    model_type=params['evaluate']['model']

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import models, transforms 

config_path = "config/evaluate_config.json"
config=utils.load_json(config_path)

model_name=config['model_name']
test_dir = os.path.join(processed_data_path,config['test_dir'])
model_path=os.path.join(output_model_path,config['model_path'])

print("test_dir: ",test_dir)
print("model: ", model_path)

augmentation = config["augmentation"]
transform = transforms.Compose([
    transforms.Resize(tuple(config["input_shape"][:2])),
    transforms.ToTensor(),
    transforms.Normalize(mean=augmentation["normalize_mean"], std=augmentation["normalize_std"])
])

test_data = datasets.ImageFolder(test_dir, transform=transform)
# class_map = {"real": 0, "fake": 1}
# test_data.class_to_idx = class_map
# test_data.targets = [class_map[os.path.basename(os.path.dirname(img_path))] for img_path, _ in test_data.samples]

for img_path, _ in test_data.samples[:5]:  # Print first 5 samples
    print(f"Image Path: {img_path} -> Extracted Label: {os.path.basename(os.path.dirname(img_path))}")

test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=False)
num_classes = len(test_data.classes)

model = CNNTrainer(config, num_classes, model_path=model_path)
image_paths, pred_labels, true_labels, test_accuracy, test_loss, results = model.evaluate_model(test_loader)

# Write statistics to CSV
header = ["model_path", "model", "tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1-score", "loss", "test_data_dir"]
stats_file_path = os.path.join(output_model_path, "cnn_model_stats.csv")
file_exists = os.path.isfile(stats_file_path)

with open(stats_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write header if the file does not exist
    if not file_exists:
        writer.writerow(header)
    # Write the statistics
    writer.writerow([config["model_path"], config["model_name"], results["tp"], results["fp"], results["tn"], results["fn"],
                    test_accuracy, results["precision"], results["recall"], results["f1-score"],  test_loss, config['test_dir']])

predictions_csv_path = os.path.join(output_model_path, "test_predictions.csv")

if os.path.exists(predictions_csv_path):
    os.remove(predictions_csv_path)

with open(predictions_csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_path", "true_label", "predicted_label"])

    # Write the data
    for img, true, pred in zip(image_paths, true_labels, pred_labels):
        writer.writerow([img, true, pred])

print(f"✅ Test predictions saved at: {predictions_csv_path}")