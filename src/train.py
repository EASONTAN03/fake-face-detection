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
import time

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import operation_utils as utils
from modeling_utils import *

timestamp = str(int(time.time()))
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
prepare_benchmark = str(params['prepare']['benchmark'])

param_train = params['train']
model_benchmark=param_train['model_benchmark']

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

data_dir = f'{dataset}_{benchmark}'
processed_data_dir = f'{dataset}_{benchmark}_{prepare_benchmark}'
print("Dataset:", processed_data_dir)

output_dir = os.path.join(output_model_path, processed_data_dir)

# Example usage
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import models, transforms 

train_dir = os.path.join(processed_data_path, data_dir, prepare_benchmark, 'train')
test_dir = os.path.join(processed_data_path, data_dir, prepare_benchmark,'test')

config_path = "config/config.json"
config=utils.load_json(config_path)

#Create save directory
model_path_dir = os.path.join(output_dir, config["model_name"])
utils.create_dir(model_path_dir)
model_time_path_dir = os.path.join(model_path_dir, timestamp)
utils.create_dir(model_time_path_dir)

# Define dataset transform using config
augmentation = config["augmentation"]
    
#explain this part
transform = transforms.Compose([
    transforms.Resize(tuple(config["input_shape"][:2])),
    transforms.RandomRotation(degrees=10 if augmentation["rotation_range"] else 0),
    transforms.RandomHorizontalFlip(p=0.5 if augmentation["horizontal_flip"] else 0),
    transforms.ToTensor(),
    transforms.Normalize(mean=augmentation["normalize_mean"], std=augmentation["normalize_std"])
])


train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=False)

val_dir = os.path.join(processed_data_path, data_dir, prepare_benchmark,'val')
val_data = datasets.ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False)


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
    "model_path": model_time_path_dir,
    "data_dir": processed_data_dir,
    "model_params": config,
}

# Path to log file
log_file_path = os.path.join(model_path_dir, "log.json")
utils.write_json(log_data, log_file_path)

image_paths, all_preds, all_labels, results = trainer.evaluate_model(test_loader)

models_dict={"model_path": model_time_path_dir}
results = {**models_dict, **results}

stats_file_path = os.path.join(output_model_path, "cnn_model_stats.csv")
file_exists = os.path.isfile(stats_file_path)

with open(stats_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(results.keys())
    writer.writerow(results.values())

predictions_csv_path = os.path.join(model_time_path_dir, "test_predictions.csv")
file_exists = os.path.isfile(predictions_csv_path)

with open(predictions_csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["image_path", "true_label", "predicted_label"])

    for img, true, pred in zip(image_paths, all_labels, all_preds):
        writer.writerow([img, true, pred])

print(f"✅ Test predictions saved at: {predictions_csv_path}")
