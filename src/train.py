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
split_ratio = params['make_dataset']['split_ratio']
prepare_benchmark = params['prepare']['benchmark']
print("Dataset:", prepare_benchmark)

param_train = params['train']
model_benchmark=param_train['model_benchmark']
model=param_train['model']
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
X_train = np.load(features_path)  # Shape: (N, 224, 224)
y_train = np.load(labels_path)  # Shape: (N,)

validate_dir= os.path.join(processed_data_path, data_dir, 'test')
validate_features_path = os.path.join(validate_dir,f'features_{prepare_benchmark}.npy')
validate_labels_path = os.path.join(validate_dir,f'labels_{prepare_benchmark}.npy')
X_test = np.load(validate_features_path)  # Shape: (N, 224, 224)
y_test = np.load(validate_labels_path)  # Shape: (N,)

model_path_dir=os.path.join(output_model_dir, f"{model_path}.pkl")

# Flatten the images (convert 224x224 into 1D arrays for each image)
print(X_train.shape)
if len(X_train.shape) > 2:
    size_1=X_train.shape[1]
    size_2=X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], size_1 * size_2)
    X_test = X_test.reshape(X_test.shape[0], size_1 * size_2)
print("Train features shape: ", X_train.shape)

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
            'C_values': model_params['C'],
            'kernels': model_params['kernel'],
            'gammas': model_params['gamma']
        }
        print(model, param_grid)
        trainer.train_svm(X_train, y_train, **param_grid, n_splits=5)
    elif model=="knn":
        param_grid = {
            'k_values': model_params['n_neighbors'],
            'weights': model_params['weights'],
            'metrics': model_params['metric']
        }
        print(model, param_grid)
        trainer.train_knn(X_train, y_train, **param_grid, n_splits=5)
    elif model=="lgbm":
        param_grid = {
            'objectives': model_params['objective'],
            'metrics': model_params['metric'],
            'num_leaves': model_params['num_leaves'],
            'learning_rates': model_params['learning_rate'],
            'max_depths': model_params['max_depth'],
            'min_leaf_list': model_params['min_data_in_leaf'],
            'training_rounds': model_params['training_round']
        }
        print(model, param_grid)
        trainer.train_lgbm(X_train, y_train, **param_grid, n_splits=5)
    elif model=="xgboost":
        param_grid = {
            'objectives': model_params['objective'],
            'learning_rates': model_params['learning_rate'],
            'max_depths': model_params['max_depth'],
            'min_child_list': model_params['min_child_weight'],
            'training_rounds': model_params['training_round']
        }
        print(model, param_grid)
        trainer.train_xgboost(X_train, y_train, **param_grid, n_splits=5)
        
    # Evaluate models
    # X_test_scaled=trainer.scaler.fit_transform(X_test)
    y_pred, results = trainer.evaluate_models(X_test, y_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()


    print(f"\n{model} Results:")
    print(f"Testing Accuracy: {results['accuracy']:.4f}")
    print(f"Testing Precision: {results['precision']:.4f}")
    print(f"Testing Loss: {results['loss']:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")



    log_data = {
        "seed": seed,
        "model_path": model_path,
        "model_benchmark": model_benchmark,
        "features_dir": processed_data_dir,
        "model_params": model_params,
        'scaler': scale_features,
        'train_test_split': split_ratio,
        "train_features_shape": X_train.shape,
        "train_labels_shape": y_train.shape,
        "output_dir": output_model_dir,
    }

    # Path to log file
    log_file_path = os.path.join(output_dir, "log.json")
    utils.write_json(log_data, log_file_path)

    # Write statistics to CSV
    header = ["model_path", "model", "tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "specificity", "f1-score", "mcc", "auroc", "loss"]
    stats_file_path = os.path.join(output_dir, "model_stats.csv")
    file_exists = os.path.isfile(stats_file_path)

    with open(stats_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if the file does not exist
        if not file_exists:
            writer.writerow(header)
        # Write the statistics
        writer.writerow([model_path, trainer.best_params, tp, fp, tn, fn, results["accuracy"], results["precision"], results["recall"], 
                        results["specificity"], results["f1-score"], results["mcc"], results["auroc"], results["loss"]])
        

    # stats_file_path = os.path.join(output_model_path, "model_stats.csv")
    # file_exists = os.path.isfile(stats_file_path)

    # with open(stats_file_path, mode='a', newline='') as file:
    #     writer = csv.writer(file)
    #     # Write header if the file does not exist
    #     if not file_exists:
    #         writer.writerow(header)
    #     # Write the statistics
    #     writer.writerow([model_path, model_params, tp, fp, tn, fn, stats["accuracy"], stats["precision"], stats["recall"], 
    #                     stats["specificity"], stats["mcc"], stats["auroc"]])
    
    print("Finish training")
    # trainer.plot_learning_curves(model,X,y)
        
