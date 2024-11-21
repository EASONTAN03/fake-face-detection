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
import lightgbm as lgb
import xgboost as xgb
    

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
        self.best_params=None
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
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
        self.best_params=str(best_model_params)
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
                        y_true_score=np.where(y_val == 1, 1, -1)
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

        self.best_params=str(best_model_params)
        # Save the best model
        model_filename = f"{self.benchmark}_best_knn_model.pkl"
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
        csv_filename = f"{self.benchmark}_{self.model}_tuning_results.csv"
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
        y_true_score=np.where(y_test == 0, -1, 1)

        # Calculate hinge loss
        if self.model=='svm':
            predictions = self.pipeline.predict(X_test)
            y_pred_scores = np.where(predictions == 1, 1, -1)
            loss = hinge_loss(y_true_score, y_pred_scores)

        elif self.model=='knn':
            predictions = self.pipeline.predict(X_test)
            y_pred_scores = np.where(predictions == 1, 1, -1)
            loss = hinge_loss(y_true_score, y_pred_scores)

        elif self.model=='lgbm':
            predictions = self.pipeline.predict(X_test)
            y_pred_scores = np.where(predictions >= 0.5, 1, -1)
            predictions=np.where(predictions >= 0.5, 1, 0)
            loss = hinge_loss(y_true_score, y_pred_scores)

        elif self.model=='xgboost':
            predictions = self.pipeline.predict(X_test)
            y_pred_scores = np.where(predictions == 1, 1, -1)
            loss = hinge_loss(y_true_score, y_pred_scores)

        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        stats = utils.compute_stats(tn, tp, fp, fn)
        stats['loss']=loss
        
        return predictions, stats
    

    def train_lgbm(self, X_train, y_train, objectives, metrics, num_leaves, learning_rates, max_depths, min_leaf_list, training_rounds, n_splits=5):
        """Train and evaluate LightGBM model using cross-validation."""
        self.random_state
        best_mcc = -1
        best_model_params = None
        best_pipeline = None
        verbose_eval=True
        print("Starting LGBM tuning with manual cross-validation...\n")
        print(f"Train size: {len(y_train)}")
        results = []

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',  # You can use other metrics as well
            'num_leaves': 45,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_data_in_leaf': 20,
            'seed': self.random_state 
        }

        # Cross-validation loop

        for training_round in training_rounds:
            # for boosting in boostings:
                for objective in objectives:
                    for metric in metrics:
                        for n_leaves in num_leaves:
                            for learning_rate in learning_rates:
                                for max_depth in max_depths:
                                    for min_data_in_leaf in min_leaf_list:
                                        params['objective']=objective
                                        params['metric']=metric
                                        params['num_leaves']=n_leaves
                                        params['learning_rate']=learning_rate
                                        params['max_depth']=max_depth
                                        params['min_data_in_leaf']=min_data_in_leaf
                                        print(f"Testing LGBM with objective='{params['objective']}', metric='{params['metric']}',"
                                            f"num_leaves={params['num_leaves']}, learning_rate={params['learning_rate']}, max_depth={params['max_depth']},"
                                            f"min_data_in_leaf={params['min_data_in_leaf']}, training_round={training_round}'")
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

                                            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

                                            model = lgb.train(params=params, train_set=lgb.Dataset(X_train_fold, label=y_train_fold), num_boost_round=training_round)
                                            predictions = model.predict(X_val) # Round for binary classification
                                            y_pred=np.where(predictions >= 0.5, 1, 0)

                                            y_pred_scores = np.where(predictions >= 0.5, 1, -1)
                                            y_true_score=np.where(y_val == 1, 1, -1)

                                            # Calculate hinge loss
                                            fold_loss = hinge_loss(y_true_score, y_pred_scores)
                                            
                                            # Confusion matrix for fold-specific TP, FP, TN, FN
                                            print(y_val, y_pred)
                                            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
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
                                            "LGBM",  # model name
                                            f"Testing LGBM with objective={params['objective']}, metric={params['metric']}," 
                                            f"num_leaves={params['num_leaves']}, learning_rate={params['learning_rate']}, max_depth={params['max_depth']},"
                                            f"min_data_in_leaf={params['min_data_in_leaf']}, training_round={training_round}",
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
                                            best_model_params = (training_round, objective, metric, n_leaves, learning_rate, max_depth, min_data_in_leaf)
                                        # if avg_f1 > best_mcc:
                                        #     best_mcc = avg_f1
                                        #     best_model_params = (C, kernel, gamma)
        # training_round=best_model_params[0]
        # params['booster']=best_model_params[1]
        # params['objective']=best_model_params[2]
        # params['metric']=best_model_params[3]
        # params['num_leaves']=best_model_params[4]
        # params['learning_rate']=best_model_params[5]
        # params['max_depth']=best_model_params[6]
        # params['min_data_in_leaf']=best_model_params[7]
    
        training_round=best_model_params[0]
        params['objective']=best_model_params[1]
        params['metric']=best_model_params[2]
        params['num_leaves']=best_model_params[3]
        params['learning_rate']=best_model_params[4]
        params['max_depth']=best_model_params[5]
        params['min_data_in_leaf']=best_model_params[6]

        model = lgb.train(params=params, train_set=lgb.Dataset(X_train, label=y_train), num_boost_round=training_round)
        self.pipeline=model   

        # Save the best model
        model_filename = f"{self.benchmark}_best_lgbm_model.pkl"
        model_path=os.path.join(self.checkpoint_dir,model_filename)
        self.best_params=str(best_model_params)
        with open(model_path, 'wb') as file:
            pickle.dump(best_pipeline, file)
        
        print(f"Best MCC: {best_mcc:.4f}")
        print(f"Best model saved with parameters: training_round={best_model_params[0]}, "
            f"n_leaves='{best_model_params[1]}', learning_rate='{best_model_params[2]}', max_depth='{best_model_params[3]}', min_data_in_leaf='{best_model_params[4]}'")
        print(f"Model saved as {model_filename}")
        
        self.save_cv_results(results)
        return model

    def train_xgboost(self, X_train, y_train, objectives, learning_rates, max_depths, min_child_list, training_rounds, n_splits=5):
        """Train and evaluate LightGBM model using cross-validation."""
        self.random_state
        best_mcc = -1
        best_model_params = None
        best_pipeline = None
        # verbose_eval=True
        print("Starting XGBoost tuning with manual cross-validation...\n")
        print(f"Train size: {len(y_train)}")
        results = []

        params = {
            'objective': 'binary:hinge',
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 5,
            'random_state': self.random_state 
        }

        for training_round in training_rounds:
            for objective in objectives:
                for learning_rate in learning_rates:
                    for max_depth in max_depths:
                        for min_child_weight in min_child_list:
                            params['objective']=objective
                            params['learning_rate']=learning_rate
                            params['max_depth']=max_depth
                            params['min_child_weight']=min_child_weight
                            print(f"Testing XGBoost with objective='{params['objective']}',"
                                f"learning_rate={params['learning_rate']}, max_depth={params['max_depth']},"
                                f"min_child_weight={params['min_child_weight']}, training_round={training_round}'")
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

                                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

                                model = xgb.XGBClassifier(**params, n_estimators=training_round)
                                model.fit(X_train_fold, y_train_fold)

                                predictions = model.predict(X_val)
                                
                                # Calculate hinge loss
                                y_pred_scores = np.where(predictions == 1, 1, -1)
                                y_true_score=np.where(y_val == 1, 1, -1)

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
                                "XGBoost",  # model name
                                f"Testing XGBoost with objective='{params['objective']}',"
                                f"learning_rate={params['learning_rate']}, max_depth={params['max_depth']},"
                                f"min_child_weight={params['min_child_weight']}, training_round={training_round}'",
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
                                best_model_params = (training_round, objective, learning_rate, max_depth, min_child_weight)
                            # if avg_f1 > best_mcc:
                            #     best_mcc = avg_f1
                            #     best_model_params = (C, kernel, gamma)
        training_round=best_model_params[0]
        params['objective']=best_model_params[1]
        params['learning_rate']=best_model_params[2]
        params['max_depth']=best_model_params[3]
        params['min_child_weight']=best_model_params[4]

        model = xgb.XGBClassifier(**params, n_estimators=training_round)
        model.fit(X_train, y_train)    
        self.pipeline=model   
        self.best_params=str(best_model_params)
        # Save the best model
        model_filename = f"{self.benchmark}_best_xgboost_model.pkl"
        model_path=os.path.join(self.checkpoint_dir,model_filename)

        with open(model_path, 'wb') as file:
            pickle.dump(best_pipeline, file)
        
        print(f"Best MCC: {best_mcc:.4f}")
        print(f"Best model saved with parameters: training_round={best_model_params[0]},"
            f"objective='{best_model_params[1]}', learning_rate='{best_model_params[2]}', max_depth='{best_model_params[3]}', min_child_weight='{best_model_params[4]}'")
        print(f"Model saved as {model_filename}")
        
        self.save_cv_results(results)
        return model
    