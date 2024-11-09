import os 
import numpy as np
import json
import csv
import pickle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer, log_loss, hinge_loss
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import random
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import operation_utils as utils
from modeling_utils import *


class ModelTrainer:
    def __init__(self, model="svm", checkpoint_dir='checkpoints', model_benchmark=1, random_state=42):
        
        self.random_state = random_state
        self.benchmark = model_benchmark
        self.model=model
        self.scaler = StandardScaler()
        self.pipeline = None
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
    def get_latest_checkpoint(self, model_type):
        """
        Get the most recent checkpoint for the specified model type
        """
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                      if f.startswith(model_type) and f.endswith('.joblib')]
        
        if not checkpoints:
            return None, None
        
        latest_model = max(checkpoints)
        latest_history = latest_model.replace('.joblib', '.json')
        
        model_path = os.path.join(self.checkpoint_dir, latest_model)
        history_path = os.path.join(self.checkpoint_dir, latest_history)
        
        return model_path, history_path
    
    def train_svm(self, X_train, y_train, C_values, kernels, gammas, n_splits):
        self.random_state
        best_mcc = -1
        best_model_params = None
        best_pipeline = None
        print("Starting SVM tuning with manual cross-validation...\n")
        print(f"Train size: {len(y_train)}")
        results = []
        
        # Cross-validation loop
        for C in C_values:
            for kernel in kernels:
                for gamma in gammas:
                    print(f"Testing SVM with C={C}, kernel='{kernel}', gamma='{gamma}'")
                    
                    # To store cross-validation scores for each metric
                    acc_scores = []
                    precision_scores = []
                    recall_scores = []
                    specificity_scores=[]
                    f1_scores = []
                    mcc_scores=[]
                    auroc_scores=[]
                    loss_scores = []
                    tp_total, fp_total, tn_total, fn_total = 0, 0, 0, 0
                    
                    # Manually split data into n_splits folds
                    fold_size = len(X_train) // n_splits
                    print(f"Number of folds: {n_splits}, Fold size: {fold_size}\n")
                    
                    # Initialize Stratified K-Fold
                    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

                    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
                        print(f"Fold {fold + 1}/{n_splits}")
                        
                        # Split the data using the indices from Stratified K-Fold
                        X_train_fold, X_val = X_train[train_index], X_train[val_index]
                        y_train_fold, y_val = y_train[train_index], y_train[val_index]


                        # Create and train SVM pipeline for this fold
                        pipeline = Pipeline([
                            ('svm', SVC(C=C, kernel=kernel, gamma=gamma, 
                                    probability=True, random_state=self.random_state))
                        ])
                        
                        # Train the model on the training fold
                        pipeline.fit(X_train_fold, y_train_fold)
                        
                        # Predict on the validation fold
                        predictions = pipeline.predict(X_val)
                        probabilities = pipeline.predict_proba(X_val)
                        
                        # Calculate hinge loss
                        fold_loss = hinge_loss(y_val, probabilities[:, 1])
                        
                        # Confusion matrix for fold-specific TP, FP, TN, FN
                        tn, fp, fn, tp = confusion_matrix(y_val, predictions).ravel()
                        stats = utils.compute_stats(tn, tp, fp, fn)

                        
                        # Append fold scores
                        acc_scores.append(stats['accuracy'])
                        precision_scores.append(stats['precision'])
                        recall_scores.append(stats['recall'])
                        specificity_scores.append(stats['specificity'])
                        f1_scores.append(stats['f1-score'])
                        mcc_scores.append(stats['mcc'])
                        auroc_scores.append(stats['auroc'])
                        loss_scores.append(fold_loss)
                        tp_total += tp
                        fp_total += fp
                        tn_total += tn
                        fn_total += fn
                        
                        print(f"    Accuracy: {stats['accuracy']:.4f}, Precision: {stats['precision']:.4f}, Recall: {stats['recall']:.4f},"
                            f"Specificity: {stats['specificity']:.4f}, F1-Score: {stats['f1-score']:.4f}"
                            f"MCC: {stats['mcc']:.4f}, AUROC: {stats['auroc']:.4f}, Loss: {fold_loss:.4f}")
                        print(f"    TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n")
                    
                    # Average scores across all folds
                    avg_acc = np.mean(acc_scores)
                    avg_precision = np.mean(precision_scores)
                    avg_recall = np.mean(recall_scores)
                    avg_specificity = np.mean(recall_scores)
                    avg_f1 = np.mean(f1_scores)
                    avg_mcc= np.mean(mcc_scores)
                    avg_auroc = np.mean(auroc_scores)
                    avg_loss = np.mean(loss_scores)
                    
                    # Append cross-validation results
                    results.append([
                        "SVM",  # model name
                        f"C={C}, kernel='{kernel}', gamma={gamma}",  # model hyperparameter details
                        len(X_train_fold),  # train size
                        len(X_val),  # validation size
                        tp_total, fp_total, tn_total, fn_total,  # total confusion matrix counts
                        avg_acc, avg_precision, avg_recall, avg_specificity, avg_f1, avg_mcc, avg_auroc, avg_loss  # averaged metrics
                    ])
                    
                    # Check if this is the best model so far based on loss
                    # if avg_loss < best_loss:
                    #     best_loss = avg_loss
                    #     best_model_params = (C, kernel, gamma)
                    if avg_mcc > best_mcc:
                        best_mcc = avg_mcc
                        best_model_params = (C, kernel, gamma)
                    # if avg_f1 > best_mcc:
                    #     best_mcc = avg_f1
                    #     best_model_params = (C, kernel, gamma)

        # Retrain the best model on the full training set
        print(best_model_params)
        best_pipeline = Pipeline([
            ('svm', SVC(C=best_model_params[0], 
                    kernel=best_model_params[1], 
                    gamma=best_model_params[2],
                    probability=True,
                    random_state=self.random_state))
        ])
        best_pipeline.fit(X_train, y_train)

        self.pipeline=best_pipeline   
        # Save the best model
        model_filename = f"{self.benchmark}_best_svm_model.pkl"
        model_path=os.path.join(self.checkpoint_dir,model_filename)

        with open(model_path, 'wb') as file:
            pickle.dump(best_pipeline, file)
        
        print(f"Best MCC: {best_mcc:.4f}")
        print(f"Best model saved with parameters: C={best_model_params[0]}, "
            f"kernel='{best_model_params[1]}', gamma='{best_model_params[2]}'")
        print(f"Model saved as {model_filename}")
        
        self.save_cv_results(results)
             

    
    def train_knn(self, X_train, y_train, k_values, weights, metrics, n_splits=5):
        """
        Train KNN model with hyperparameter tuning
        """
        self.random_state
        best_mcc = -1
        best_model_params = None
        best_pipeline = None
        print("Starting KNN tuning with manual cross-validation...\n")
        print(f"Train size: {len(y_train)}")
        results = []
        
        # Cross-validation loop
        for k in k_values:
            for weight in weights:
                for metric in metrics:
                    print(f"Testing KNN with k={k}, weights='{weight}', metric='{metric}'")
                    
                    # To store cross-validation scores for each metric
                    acc_scores = []
                    precision_scores = []
                    recall_scores = []
                    specificity_scores=[]
                    f1_scores = []
                    mcc_scores=[]
                    auroc_scores=[]
                    loss_scores = []
                    tp_total, fp_total, tn_total, fn_total = 0, 0, 0, 0
                    
                    # Manually split data into n_splits folds
                    fold_size = len(X_train) // n_splits
                    print(f"Number of folds: {n_splits}, Fold size: {fold_size}\n")
                    
                    # Initialize Stratified K-Fold
                    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

                    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
                        print(f"Fold {fold + 1}/{n_splits}")
                        
                        # Split the data using the indices from Stratified K-Fold
                        X_train_fold, X_val = X_train[train_index], X_train[val_index]
                        y_train_fold, y_val = y_train[train_index], y_train[val_index]

                        # Create and train SVM pipeline for this fold
                        pipeline = Pipeline([
                            ('knn', KNeighborsClassifier(n_neighbors=k, weights=weight, metric=metric, n_jobs=-1))
                        ])
                        
                        # Train the model on the training fold
                        pipeline.fit(X_train_fold, y_train_fold)
                        
                        # Predict on the validation fold
                        predictions = pipeline.predict(X_val)
                        
                        # Calculate hinge loss
                        y_true_score=np.where(y_val == 0, -1, 1)
                        y_pred_scores = np.where(predictions == 1, 1, -1)

                        # Calculate hinge loss
                        fold_loss = hinge_loss(y_true_score, y_pred_scores)
                        
                        # Confusion matrix for fold-specific TP, FP, TN, FN
                        tn, fp, fn, tp = confusion_matrix(y_val, predictions).ravel()
                        stats = utils.compute_stats(tn, tp, fp, fn)

                        
                        # Append fold scores
                        acc_scores.append(stats['accuracy'])
                        precision_scores.append(stats['precision'])
                        recall_scores.append(stats['recall'])
                        specificity_scores.append(stats['specificity'])
                        f1_scores.append(stats['f1-score'])
                        mcc_scores.append(stats['mcc'])
                        auroc_scores.append(stats['auroc'])
                        loss_scores.append(fold_loss)
                        tp_total += tp
                        fp_total += fp
                        tn_total += tn
                        fn_total += fn
                        
                        print(f"    Accuracy: {stats['accuracy']:.4f}, Precision: {stats['precision']:.4f}, Recall: {stats['recall']:.4f},"
                            f"Specificity: {stats['specificity']:.4f}, F1-Score: {stats['f1-score']:.4f}"
                            f"MCC: {stats['mcc']:.4f}, AUROC: {stats['auroc']:.4f}, Loss: {fold_loss:.4f}")
                        print(f"    TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n")
                    
                    # Average scores across all folds
                    avg_acc = np.mean(acc_scores)
                    avg_precision = np.mean(precision_scores)
                    avg_recall = np.mean(recall_scores)
                    avg_specificity = np.mean(recall_scores)
                    avg_f1 = np.mean(f1_scores)
                    avg_mcc= np.mean(mcc_scores)
                    avg_auroc = np.mean(auroc_scores)
                    avg_loss = np.mean(loss_scores)
                    
                    # Append cross-validation results
                    results.append([
                        "KNN",  # model name
                        f"k={k}, weights='{weight}', metric='{metric}",  # model hyperparameter details
                        len(X_train_fold),  # train size
                        len(X_val),  # validation size
                        tp_total, fp_total, tn_total, fn_total,  # total confusion matrix counts
                        avg_acc, avg_precision, avg_recall, avg_specificity, avg_f1, avg_mcc, avg_auroc, avg_loss  # averaged metrics
                    ])
                    
                    # Check if this is the best model so far based on loss
                    # if avg_loss < best_loss:
                    #     best_loss = avg_loss
                    #     best_model_params = (C, kernel, gamma)
                    if avg_mcc > best_mcc:
                        best_mcc = avg_mcc
                        best_model_params = (k, weight, metric)
                    # if avg_f1 > best_mcc:
                    #     best_mcc = avg_f1
                    #     best_model_params = (C, kernel, gamma)

        # Retrain the best model on the full training set
        print(best_model_params)
        best_pipeline = Pipeline([
            ('knn', KNeighborsClassifier(n_neighbors=best_model_params[0], 
                    weights=best_model_params[1], 
                    metric=best_model_params[2],
                    n_jobs=-1))
        ])
        best_pipeline.fit(X_train, y_train)

        self.pipeline=best_pipeline   
        # Save the best model
        model_filename = f"{self.benchmark}_best_svm_model.pkl"
        model_path=os.path.join(self.checkpoint_dir,model_filename)

        with open(model_path, 'wb') as file:
            pickle.dump(best_pipeline, file)
        
        print(f"Best MCC: {best_mcc:.4f}")
        print(f"Best model saved with parameters: k={best_model_params[0]}, "
            f"weight='{best_model_params[1]}', metric='{best_model_params[2]}'")
        print(f"Model saved as {model_filename}")
        
        self.save_cv_results(results)
    

    def save_cv_results(self, results):
        # Save results to CSV
        csv_filename = f"{self.benchmark}_svm_tuning_results.csv"
        csv_path=os.path.join(self.checkpoint_dir,csv_filename)
        
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Model Name", "Hyperparameter", "Train Size", "Validation Size", 
                "TP", "FP", "TN", "FN", "Accuracy", "Precision", "Recall", 
                "Specificity", "F1-Score", "MCC", "AUROC", "Hinge Loss"
            ])
            writer.writerows(results)
            print(f"Cross Validation Results saved to {csv_filename}")
        print("Saved Cross Validation Results to CSV....")
        
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate both models 
        """
        predictions=self.pipeline.predict(X_test)
    
        # Calculate hinge loss
        if self.model=='svm':
            probabilities = self.pipeline.predict_proba(X_test)
            loss = hinge_loss(y_test, probabilities[:, 1])
        elif self.model=='knn':
            y_true_test  = np.where(y_test == 0, -1, 1)
            y_pred_scores = np.where(predictions == 1, 1, -1)
            loss = hinge_loss(y_true_test, y_pred_scores)

        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        stats = utils.compute_stats(tn, tp, fp, fn)
        stats['loss']=loss
        
        return predictions, stats
    
    def plot_learning_curves(self, model, X, y, cv=5):
        """
        Plot learning curves for both models
        """
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        plt.figure(figsize=(12, 4))
        
        # Plot SVM learning curve if model exists
        if model=='svm' and self.best_svm is not None:
            plt.subplot(1, 2, 1)
            train_sizes, train_scores, test_scores = learning_curve(
                self.best_svm, X, y, cv=cv, n_jobs=-1, 
                train_sizes=train_sizes, scoring='neg_log_loss'
            )
            
            train_mean = -np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = -np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            plt.plot(train_sizes, train_mean, label='Training score')
            plt.plot(train_sizes, test_mean, label='Cross-validation score')
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
            plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
            plt.xlabel('Training Examples')
            plt.ylabel('Loss')
            plt.title(f'SVM Learning Curve\nIterations: {self.training_history["svm"]["iterations"]}')
            plt.legend(loc='best')
        
        # Plot KNN learning curve if model exists
        if model=='knn' and self.best_knn is not None:
            plt.subplot(1, 2, 2)
            train_sizes, train_scores, test_scores = learning_curve(
                self.best_knn, X, y, cv=cv, n_jobs=-1, 
                train_sizes=train_sizes, scoring='neg_log_loss'
            )
            
            train_mean = -np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = -np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            plt.plot(train_sizes, train_mean, label='Training score')
            plt.plot(train_sizes, test_mean, label='Cross-validation score')
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
            plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
            plt.xlabel('Training Examples')
            plt.ylabel('Loss')
            plt.title(f'KNN Learning Curve\nIterations: {self.training_history["knn"]["iterations"]}')
            plt.legend(loc='best')
        
        plt.tight_layout()
        plt.show()