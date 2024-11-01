import yaml
import csv
import os 
import cv2
import numpy as np
import pandas as pd
import json
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer, log_loss
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import random
import matplotlib.pyplot as plt
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import operation_utils as utils
from modeling_utils import *

current_dir = os.getcwd()
print(current_dir)
base_path = os.path.abspath(os.path.join(current_dir, '..'))
print("Base Path:", base_path)

with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

processed_data_path = os.path.join(base_path,config['dataset']['processed'])
output_model_path = os.path.join(base_path,config['output']['models'])
dataset = config['configs']['dataset']
benchmark = config['configs']['benchmark']

with open('../params.yaml', 'r') as file:
    params = yaml.safe_load(file)
seed = params['make_dataset']['seed']
prepare_benchmark = params['prepare']['benchmark']

param_train = params['train']
continue_training = param_train['continue_training']
model_benchmark=param_train['model_benchmark']
model=param_train['model']
split_ratio=param_train['split_ratio']
scale_features=param_train['scaler']
model_params = param_train[f"{model}"]

np.random.seed(seed)
random.seed(seed)

data_dir = f'{dataset}_{benchmark}'
processed_data_dir = f'{dataset}_{benchmark}_{prepare_benchmark}'
output_dir = os.path.join(output_model_path, processed_data_dir)
utils.create_dir(output_dir)
output_model_dir = os.path.join(output_dir, model)
utils.create_dir(output_model_dir)
model_path = f'{dataset}_{benchmark}_{prepare_benchmark}_{model_benchmark}'

input_dir = os.path.join(processed_data_path, data_dir, 'train')
features_path = os.path.join(input_dir,f'features_{prepare_benchmark}.npy')
labels_path = os.path.join(input_dir,f'labels_{prepare_benchmark}.npy')
features = np.load(features_path)  # Shape: (N, 224, 224)
y_train = np.load(labels_path)  # Shape: (N,)

validate_dir= os.path.join(processed_data_path, data_dir, 'test')
validate_features_path = os.path.join(validate_dir,f'features_{prepare_benchmark}.npy')
validate_labels_path = os.path.join(validate_dir,f'labels_{prepare_benchmark}.npy')
X_test = np.load(validate_features_path)  # Shape: (N, 224, 224)
y_test = np.load(validate_labels_path)  # Shape: (N,)

model_path_dir=os.path.join(output_model_dir, f"{model_path}.pkl")

# Flatten the images (convert 224x224 into 1D arrays for each image)
size_1=features.shape[1]
size_2=features.shape[2]
X_train = features.reshape(features.shape[0], size_1 * size_2)
X_test = features.reshape(features.shape[0], size_1 * size_2)
print("Features shape: ", features.shape)

X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)
 
# Example usage
if __name__ == "__main__":    
    # Initialize trainer
    trainer = ModelTrainer(model=model, checkpoint_dir=output_model_dir, model_benchmark=model_benchmark,random_state=seed)

    # Train models
    # First training session
    if model=="svm":
        param_grid = {
            'svm__C': model_params['C'],
            'svm__kernel': model_params['kernel'],
            'svm__gamma': model_params['gamma']
        }
        
        trainer.train_svm(X_train, y_train, param_grid, continue_training)
    elif model=="knn":
        param_grid = {
            'knn__n_neighbors': model_params['n_neighbors'],
            'knn__weights': model_params['weights'],
            'knn__metric': model_params['metric']
        }
        trainer.train_knn(X_train, y_train, param_grid, continue_training)
    
    # Evaluate models
    X_test_scaled=trainer.scaler.transform(X_test)
    y_pred, results = trainer.evaluate_models(X_test_scaled, y_test)
    
    # Print results
    print("\nFirst Training Session Results:")   
    print(f"\n{model} Results:")
    print(f"Accuracy: {results[model]['accuracy']:.4f}")
    print(f"Log Loss: {results[model]['log_loss']:.4f}")
    print("Best Parameters:", results[model]['best_params'])


    test_accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion.ravel()
    stats = utils.compute_stats(tn, tp, fp, fn)

    print(f"\n{model} Results:")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    print(f"Confusion Matrix:\n{confusion}")


    model_params=f"model:{model}, {utils.dict2str(results[model]['best_params'])}"
    if model_params.endswith(',  '):
        model_params = model_params[:-3]  

    log_data = {
        "seed": seed,
        "model_path": model_path,
        "model_benchmark": model_benchmark,
        "features_dir": processed_data_dir,
        "model_params": model_params,
        'scaler': scale_features,
        'train_test_split': split_ratio,
        "train_features_shape": features.shape,
        "train_labels_shape": y_train.shape,
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
    
    
    trainer.plot_learning_curves(X,y)
        
