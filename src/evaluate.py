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
import modeling_utils as modeling
current_dir = os.getcwd()

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

processed_data_path = config['dataset']['processed']
output_model_path = config['output']['models']
dataset = config['configs']['dataset']
benchmark = config['configs']['benchmark']

with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)
seed = params['make_dataset']['seed']
prepare_benchmark = params['prepare']['benchmark']
param_train = params['train']
model_benchmark=param_train['model_benchmark']
model=param_train['model']
param_model = params['train'][f"{model}"]
split_ratio=param_train['split_ratio']
scale_features=param_train['scale_features']
train=param_train['train']


np.random.seed(seed)
random.seed(seed)

data_dir = f'{dataset}_{benchmark}'
processed_data_dir = f'{dataset}_{benchmark}_{prepare_benchmark}'
output_dir = os.path.join(output_model_path, processed_data_dir)
utils.create_dir(output_dir)
output_model_dir = os.path.join(output_dir, model)
utils.create_dir(output_model_dir)
model_path = f'{dataset}_{benchmark}_{prepare_benchmark}_{model_benchmark}'

input_dir = os.path.join(processed_data_path, data_dir, 'test')
features_path = os.path.join(input_dir,f'features_{prepare_benchmark}.npy')
labels_path = os.path.join(input_dir,f'labels_{prepare_benchmark}.npy')
features = np.load(features_path)  # Shape: (N, 224, 224, 3)
labels = np.load(labels_path)  # Shape: (N,)

model_path_dir=os.path.join(output_model_dir, f"{model_path}.pkl")

# Flatten the images (convert 224x224x3 into 1D arrays for each image)
size_1=features.shape[1]
size_2=features.shape[2]
if features.shape[-1] == 3:
    features = features.reshape(features.shape[0], size_1 * size_2 * 3)
else:
    features = features.reshape(features.shape[0], size_1 * size_2)
print("Features shape: ", features.shape)

X_test=features
y_test=labels

if scale_features =="standard":
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
else:
    X_test_scaled = X_test

# Load the trained model
trained_model = joblib.load(model_path_dir)
parameters=None

test_accuracy = accuracy_score(y_test, trained_model.predict(X_test_scaled))
confusion = confusion_matrix(y_test, trained_model.predict(X_test_scaled))
tn, fp, fn, tp = confusion.ravel()
stats = utils.compute_stats(tn, tp, fp, fn)

print(f"\n{model} Results:")
print(f"Testing Accuracy: {test_accuracy:.4f}")
print(f"Confusion Matrix:\n{confusion}")


model_params=f"model:{model}, {utils.dict2str(parameters)}"
if model_params.endswith(',  '):
    model_params = model_params[:-3]  

log_data = {
    "model_path": model_path,
    "model_benchmark": model_benchmark,
    "features_dir": processed_data_dir,
    "model_params": model_params,
    "seed": seed,
    'train_test_split': split_ratio,
    'scaler': scale_features,
    "features_shape": features.shape,
    "labels_shape": labels.shape,
    "output_dir": output_model_dir,
}

# Path to log file
log_file_path = os.path.join(output_dir, "log.json")
utils.write_json(log_data, log_file_path)

# Write statistics to CSV
header = ["model_path", "model", "tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "specificity", "mcc", "auroc"]
stats_file_path = os.path.join(output_dir, "model_stats.csv")
file_exists = os.path.isfile(stats_file_path)

with open(stats_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write header if the file does not exist
    if not file_exists:
        writer.writerow(header)
    # Write the statistics
    writer.writerow([model_path, model_params, tp, fp, tn, fn, stats["accuracy"], stats["precision"], stats["recall"], 
                     stats["specificity"], stats["mcc"], stats["auroc"]])
    

stats_file_path = os.path.join(output_model_path, "model_stats.csv")
file_exists = os.path.isfile(stats_file_path)

with open(stats_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write header if the file does not exist
    if not file_exists:
        writer.writerow(header)
    # Write the statistics
    writer.writerow([model_path, model_params, tp, fp, tn, fn, stats["accuracy"], stats["precision"], stats["recall"], 
                     stats["specificity"], stats["mcc"], stats["auroc"]])