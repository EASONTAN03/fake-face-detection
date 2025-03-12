import yaml
import csv
import os 
import cv2
import numpy as np
import pandas as pd
import json
from sklearn.metrics import confusion_matrix

import random
import matplotlib.pyplot as plt
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import operation_utils as utils
from modeling_utils import *

base_path = os.path.abspath(os.path.join(os.getcwd(), '..'))

with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

processed_data_path = os.path.join(base_path,config['dataset']['processed'])
output_model_path = os.path.join(base_path,config['output']['models'])
dataset = config['configs']['dataset']
benchmark = config['configs']['benchmark']

with open('../params.yaml', 'r') as file:
    params = yaml.safe_load(file)
seed = params['make_dataset']['seed']
split_ratio = params['make_dataset']['split_ratio']
prepare_benchmark = str(params['prepare']['benchmark'])

param_train = params['train']
model_benchmark=param_train['model_benchmark']
model=param_train['model']

np.random.seed(seed)
random.seed(seed)

data_dir = f'{dataset}_{benchmark}'
processed_data_dir = f'{dataset}_{benchmark}_{prepare_benchmark}'
print("Dataset:", processed_data_dir)

output_dir = os.path.join(output_model_path, processed_data_dir)
utils.create_dir(output_dir)
output_model_dir = os.path.join(output_dir, model)
utils.create_dir(output_model_dir)

# Example usage
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import models, transforms 

train_dir = os.path.join(processed_data_path, data_dir, prepare_benchmark, 'train')
val_dir = os.path.join(processed_data_path, data_dir, prepare_benchmark, 'val')
test_dir = os.path.join(processed_data_path, data_dir, prepare_benchmark,'test')

config_path = "config/config.json"
config=utils.load_json(config_path)

#Create save directory
model_path_dir = os.path.join(output_model_dir, config["model_name"])
utils.create_dir(model_path_dir)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
model_time_path_dir = os.path.join(model_path_dir, timestamp)
utils.create_dir(model_time_path_dir)

# Define dataset transform using config
augmentation = config["augmentation"]
if config["input_shape"][2] == 1:
    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(tuple(config["input_shape"][:2])),
    # transforms.RandomRotation(augmentation["rotation_range"]),
    # transforms.RandomHorizontalFlip(p=0.5 if augmentation["horizontal_flip"] else 0),
    transforms.ToTensor(),
    transforms.Normalize(mean=augmentation["normalize_mean"][0], std=augmentation["normalize_std"][0])
])
    
#explain this part
else:
    transform = transforms.Compose([
        transforms.Resize(tuple(config["input_shape"][:2])),
        # transforms.RandomRotation(augmentation["rotation_range"]),
        # transforms.RandomHorizontalFlip(p=0.5 if augmentation["horizontal_flip"] else 0),
        transforms.ToTensor(),
        transforms.Normalize(mean=augmentation["normalize_mean"], std=augmentation["normalize_std"])
    ])


train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=False)

# for images, labels in train_loader:  # Replace `train_loader` with your actual DataLoader
#     print(f"Image Batch Shape: {images.shape}")  # Should be (batch_size, channels, height, width)
#     print(f"Label Batch Shape: {labels.shape}")  # Should be (batch_size,)
#     break  # Print only once

num_classes = len(train_data.classes)
trainer = CNNTrainer(config, num_classes, model_time_path_dir)

history = trainer.train(train_loader, val_loader)

try:
    trainer.save_model()
except Exception as e:
    print(f"Error saving model: {e}")

try:
    trainer.save_training_history(history)
except Exception as e:
    print(f"Error saving training history: {e}")

log_data = {
    "seed": seed,
    "model_benchmark": model_benchmark,
    "model_path": model_time_path_dir,
    "data_dir": processed_data_dir,
    "model_params": config,
}

# Path to log file
log_file_path = os.path.join(model_path_dir, "log.json")
utils.write_json(log_data, log_file_path)

image_paths, pred_labels, true_labels, test_accuracy, test_loss, results = trainer.evaluate_model(test_loader)

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
    writer.writerow([model_time_path_dir, config["model_name"], results["tp"], results["fp"], results["tn"], results["fn"],
                    test_accuracy, results["precision"], results["recall"], results["f1-score"], test_loss])

# Save predictions to CSV
predictions_csv_path = os.path.join(model_time_path_dir, "test_predictions.csv")
file_exists = os.path.isfile(predictions_csv_path)

with open(predictions_csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write header if the file does not exist
    if not file_exists:
        writer.writerow(["image_path", "true_label", "predicted_label"])

    # Write the data
    for img, true, pred in zip(image_paths, true_labels, pred_labels):
        writer.writerow([img, true, pred])

print(f"✅ Test predictions saved at: {predictions_csv_path}")
