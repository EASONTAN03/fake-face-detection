import os
import csv
import cv2
import numpy as np
import json
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import operation_utils as utils
from modeling_utils import CNNTrainer

# Set base path
base_path = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Load configuration files
with open('../config.yaml', 'r') as file:
    config_yaml = yaml.safe_load(file)

with open('../params.yaml', 'r') as file:
    params = yaml.safe_load(file)
    model_type = params['evaluate']['model']

# Load evaluation-specific config
config_path = "config/evaluate_config.json"
config = utils.load_json(config_path)

# Paths
processed_data_path = os.path.join(base_path, config_yaml['dataset']['processed'])
output_model_path = os.path.join(base_path, config_yaml['output']['models'])

model_name = config['model_name']
test_dir = os.path.join(processed_data_path, config['test_dir'])
model_path = os.path.join(output_model_path, config['model_path'])

print("Test Directory:", test_dir)
print("Model Path:", model_path)

# Define image transformations
augmentation = config["augmentation"]
transform = transforms.Compose([
    transforms.Resize(tuple(config["input_shape"][:2])),
    transforms.ToTensor(),
    transforms.Normalize(mean=augmentation["normalize_mean"], std=augmentation["normalize_std"])
])

# Load data (single image or folder)
def load_data(input_path, transform):
    if os.path.isfile(input_path):  # Single image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image at {input_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor, [input_path], None
    elif os.path.isdir(input_path):  # Folder
        dataset = datasets.ImageFolder(input_path, transform=transform)
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
        return loader, [sample[0] for sample in dataset.samples], dataset.targets
    else:
        raise ValueError(f"Input path {input_path} is neither a file nor a directory")

input_path = test_dir
data, image_paths, true_labels = load_data(input_path, transform)

# Load model
num_classes = 2  # Binary classification (real/fake)
model = CNNTrainer(config, num_classes, model_path=model_path)

# Evaluate
if isinstance(data, torch.Tensor):  # Single image
    model.model.eval()
    with torch.no_grad():
        output = model.model(data.to(model.device))
        pred_label = torch.argmax(output, dim=1).item()
    pred_labels = [pred_label]
    true_labels = None
    results = {}
else:  # Folder
    image_paths, pred_labels, true_labels, results = model.evaluate_model(data)
    test_accuracy = results['accuracy']
    test_loss = results['loss']

# Save predictions to CSV
predictions_csv_path = os.path.join(output_model_path, "test_predictions.csv")
if os.path.exists(predictions_csv_path):
    os.remove(predictions_csv_path)

with open(predictions_csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_path", "true_label", "predicted_label"])
    for img_path, true, pred in zip(image_paths, true_labels if true_labels is not None else [None]*len(pred_labels), pred_labels):
        writer.writerow([img_path, true, pred])

print(f"✅ Test predictions saved at: {predictions_csv_path}")

# Save stats to CSV (only for folder evaluation)
if true_labels is not None:
    stats_file_path = os.path.join(output_model_path, "cnn_model_stats.csv")
    header = ["model_path", "model", "tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1-score", "loss", "auc_roc", "inference_time_ms", "flops_giga", "test_data_dir"]
    file_exists = os.path.isfile(stats_file_path)

    with open(stats_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([
            config["model_path"], config["model_name"], results["tp"], results["fp"], results["tn"], results["fn"],
            results["accuracy"], results["precision"], results["recall"], results["f1-score"], results["loss"],
            results["auc_roc"], results["inference_time_ms"], results["flops_giga"], config['test_dir']
        ])

# Output results
if true_labels is None:
    print(f"Prediction for {input_path}: {'fake' if pred_labels[0] == 1 else 'real'}")
else:
    print(f"Accuracy: {test_accuracy:.4f}, Loss: {test_loss:.4f}, AUC-ROC: {results['auc_roc']:.4f}")