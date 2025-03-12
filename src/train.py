import yaml
import csv
import os 
import cv2
import numpy as np
import pandas as pd
import json
from sklearn.metrics import confusion_matrix

# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer, log_loss
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline

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
if model == "cnn":
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
    header = ["model_path", "model", "tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1-score", "mcc", "loss", "test_data_dir"]
    stats_file_path = os.path.join(output_model_path, "cnn_model_stats.csv")
    file_exists = os.path.isfile(stats_file_path)

    with open(stats_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if the file does not exist
        if not file_exists:
            writer.writerow(header)
        # Write the statistics
        writer.writerow([model_time_path_dir, config["model_name"], results["tp"], results["fp"], results["tn"], results["fn"],
                        test_accuracy, results["precision"], results["recall"], results["f1-score"],  results["mcc"], test_loss])
    
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

else:  
    model_path = f'{dataset}_{benchmark}_{prepare_benchmark}_{model_benchmark}'
    scale_features=param_train['scaler']
    model_params = param_train[f"{model}"]

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

    # Initialize trainer
    print(X_train.shape)
    if len(X_train.shape) > 2:
        size_1=X_train.shape[1]
        size_2=X_train.shape[2]
        X_train = X_train.reshape(X_train.shape[0], size_1 * size_2)
        X_test = X_test.reshape(X_test.shape[0], size_1 * size_2)
    print("Train features shape: ", X_train.shape)

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    trainer = ModelTrainer(model=model, output_model_dir=output_model_dir, model_benchmark=model_benchmark, random_state=seed)

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
        
