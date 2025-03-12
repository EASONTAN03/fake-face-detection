import os 
import numpy as np
import json
import csv
import pickle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, hinge_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer, log_loss, hinge_loss
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve, StratifiedKFold
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
    def __init__(self, model="svm", output_model_dir='model', model_benchmark=1, random_state=42):
        
        self.random_state = random_state
        self.benchmark = model_benchmark
        self.model=model
        self.scaler = StandardScaler()
        self.pipeline = None
        self.output_model_dir = output_model_dir
        self.best_params=None
        utils.create_dir(output_model_dir)
    
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
        model_path=os.path.join(self.output_model_dir,model_filename)

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
        model_path=os.path.join(self.output_model_dir,model_filename)

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
        csv_path=os.path.join(self.output_model_dir,csv_filename)
        
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
        import lightgbm as lgb

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
        model_path=os.path.join(self.output_model_dir,model_filename)
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
        import xgboost as xgb

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
        model_path=os.path.join(self.output_model_dir,model_filename)

        with open(model_path, 'wb') as file:
            pickle.dump(best_pipeline, file)
        
        print(f"Best MCC: {best_mcc:.4f}")
        print(f"Best model saved with parameters: training_round={best_model_params[0]},"
            f"objective='{best_model_params[1]}', learning_rate='{best_model_params[2]}', max_depth='{best_model_params[3]}', min_child_weight='{best_model_params[4]}'")
        print(f"Model saved as {model_filename}")
        
        self.save_cv_results(results)
        return model
    
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchsummary import summary
import json
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

class BaseCNN(nn.Module):
    """
    Base class for CNN architectures. Each specific CNN (Xception, ResNet, VGG, etc.)
    should inherit from this class.
    """
    def __init__(self, num_classes):
        super(BaseCNN, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")

class MobileNetCNN(BaseCNN):
    def __init__(self, channel, num_classes):
        super(MobileNetCNN, self).__init__(num_classes)
        self.model = models.mobilenet_v2(pretrained=True)

        if channel==1:
            # Modify first layer to accept 1-channel input instead of 3
            in_features = self.model.features[0][0].in_channels  # Get original input channels
            self.model.features[0][0] = nn.Conv2d(1, in_features, kernel_size=3, stride=2, padding=1, bias=False)

        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
    
    def forward(self, x):
        return self.model(x)

class ResNetCNN(BaseCNN):
    def __init__(self, channel, num_classes):
        super(ResNetCNN, self).__init__(num_classes)
        self.model = models.resnet50(pretrained=True)

        if channel==1:
            # Modify first layer to accept 1-channel input instead of 3
            in_features = self.model.features[0][0].in_channels  # Get original input channels
            self.model.features[0][0] = nn.Conv2d(1, in_features, kernel_size=3, stride=2, padding=1, bias=False)

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

class VGGCNN(BaseCNN):
    def __init__(self, channel, num_classes):
        super(VGGCNN, self).__init__(num_classes)
        self.model = models.vgg16(pretrained=True)
        
        if channel==1:
            # Modify first layer to accept 1-channel input instead of 3
            in_features = self.model.features[0][0].in_channels  # Get original input channels
            self.model.features[0][0] = nn.Conv2d(1, in_features, kernel_size=3, stride=2, padding=1, bias=False)

        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

class XceptionCNN(BaseCNN):
    def __init__(self, channel, num_classes):
        super(XceptionCNN, self).__init__(num_classes)
        self.model = models.efficientnet_b0(pretrained=True)  # Using EfficientNet as an Xception alternative
        
        if channel==1:
            # Modify first layer to accept 1-channel input instead of 3
            first_conv_layer = self.model.features[0][0] # Get original input channels
            self.model.features[0][0] = nn.Conv2d(1, first_conv_layer.out_channels, kernel_size=3, stride=2, padding=1, bias=False)

        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)    
    

class CNNTrainer:
    """
    A structured class for training CNN models using PyTorch.
    """
    def __init__(self, config, num_classes, output_model_dir=None, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.config = config

        """
        Initializes training with a model class and JSON config.
        """
        MODEL_MAPPING = {
            "BaseCNN": BaseCNN,
            "Xception": XceptionCNN,
            "ResNet50": ResNetCNN,
            "MobileNetV2": MobileNetCNN,
            "VGG16": VGGCNN
        }

        model_name=  MODEL_MAPPING.get(self.config["model_name"])
        print(model_name)

        if model_name is None:
            raise ValueError(f"Invalid model name '{self.config['model_name']}' in config file.")
        self.model = model_name(channel=self.config["input_shape"][2], num_classes=num_classes).to(self.device)

        self.loss_function = self.get_loss_function()

        if model_path is not None:
            try:
                # Initialize the model architecture (must match the saved model)
                self.model = MODEL_MAPPING[self.config["model_name"]](num_classes).to(device)

                # Load weights into the initialized model
                state_dict = torch.load(model_path, map_location=device)
                self.model.load_state_dict(state_dict)  # Correct way to load weights
                self.model.eval()  # Set model to evaluation mode

                print(f"✅ Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
        else:
            self.optimizer = self.get_optimizer()
            self.output_model_dir = output_model_dir
            os.makedirs(output_model_dir, exist_ok=True)
            
            # Callback tracking
            # Best checkpoint tracking
            self.best_val_acc = 0.0  
            self.best_val_loss = float('inf')
            self.early_stopping_counter = 0

        # Print model summary after initializing self.model
        print("Model Summary:")
        # print(self.config["input_shape"][2])
        summary(self.model, input_size=(self.config["input_shape"][2], self.config["input_shape"][0], self.config["input_shape"][1]))
       

    def get_optimizer(self):
        """Returns the optimizer defined in the JSON config."""
        optimizers = {
            "adam": optim.Adam(self.model.parameters(), lr=self.config["learning_rate"]),
            "sgd": optim.SGD(self.model.parameters(), lr=self.config["learning_rate"], momentum=0.9),
            "rmsprop": optim.RMSprop(self.model.parameters(), lr=self.config["learning_rate"])
        }
        return optimizers.get(self.config["optimizer"], optim.Adam(self.model.parameters(), lr=0.001))

    def get_loss_function(self):
        """Returns the loss function specified in the JSON config."""
        loss_functions = {
            "cross_entropy": nn.CrossEntropyLoss(),
            "mse": nn.MSELoss()
        }
        return loss_functions.get(self.config["loss_function"], nn.CrossEntropyLoss())

    def save_checkpoint(self, epoch, val_loss, val_acc, filename="last_checkpoint.pth", best=False):
        """Saves a checkpoint including model state, optimizer state, and epoch number."""
        checkpoint_path = os.path.join(self.output_model_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
        }, checkpoint_path)
        print(f"Checkpoint saved at: {checkpoint_path}")

        if best:
            best_model_path = os.path.join(self.output_model_dir, "best_model.pth")
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Best model saved at: {best_model_path}")

    def load_checkpoint(self, checkpoint_path):
        """Loads a checkpoint to resume training."""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            self.best_val_acc = checkpoint['val_acc']
            self.best_val_loss = checkpoint['val_loss']
            print(f"Checkpoint loaded. Resuming from epoch {start_epoch}, best val accuracy: {self.best_val_acc:.4f}, best val loss: {self.best_val_loss:.4f}")
            return start_epoch
        else:
            print("No checkpoint found. Starting from scratch.")
            return 0
       
    def train(self, train_loader, val_loader, resume_checkpoint=None):
        """Trains the model using the training and validation datasets, saving checkpoints."""
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        
        start_epoch = 0
        if resume_checkpoint:
            start_epoch = self.load_checkpoint(resume_checkpoint)

        for epoch in range(start_epoch, self.config["epochs"]):
            self.model.train()
            train_loss, correct_train = 0.0, 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                correct_train += (outputs.argmax(1) == labels).sum().item()

            val_loss, correct_val = 0.0, 0
            self.model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.loss_function(outputs, labels)
                    val_loss += loss.item()
                    correct_val += (outputs.argmax(1) == labels).sum().item()

            train_acc = correct_train / len(train_loader.dataset)
            val_acc = correct_val / len(val_loader.dataset)
            history["train_loss"].append(train_loss / len(train_loader))
            history["val_loss"].append(val_loss / len(val_loader))
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            print(f"Epoch {epoch+1}/{self.config['epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            # Save last checkpoint
            self.save_checkpoint(epoch, val_loss, val_acc)

            # Save best model based on validation accuracy
            
            if self.config["callbacks"]["early_stopping"]["metric"] == "val_acc":
                # Early Stopping
                if val_acc < self.best_val_acc:
                    self.early_stopping_counter += 1
                elif val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.early_stopping_counter = 0  # Reset counter
                    self.save_checkpoint(epoch, val_loss, val_acc, filename="best_checkpoint.pth", best=True)
                else: 
                    self.early_stopping_counter = 0  # Reset counter

            elif self.config["callbacks"]["early_stopping"]["metric"] == "val_loss":
                # Early Stopping
                if val_loss > self.best_val_loss:
                    self.early_stopping_counter += 1
                elif val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0  # Reset counter
                    self.save_checkpoint(epoch, val_loss, val_acc, filename="best_checkpoint.pth", best=True)
                else: 
                    self.early_stopping_counter = 0  # Reset counter

            if self.early_stopping_counter >= self.config["callbacks"]["early_stopping"]["patience"]:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

            # Reduce LR on Plateau
            if self.early_stopping_counter >= self.config["callbacks"]["reduce_lr"]["patience"]:
                print("lr patience reached")
                old_lr = self.optimizer.param_groups[0]['lr']
                self.optimizer.param_groups[0]['lr'] *= self.config["callbacks"]["reduce_lr"]["factor"]
                print(f"Learning rate reduced from {old_lr} to {self.optimizer.param_groups[0]['lr']}")

        return history
    
    def evaluate_model(self, testloader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        image_paths = []  # Store test image paths

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(testloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())  # Convert tensor to NumPy array
                all_labels.extend(labels.cpu().numpy())  # Convert tensor to NumPy array

                # Get image paths from dataset
                batch_paths = [testloader.dataset.imgs[i][0] for i in range(batch_idx * testloader.batch_size, 
                                                                            batch_idx * testloader.batch_size + len(labels))]
                image_paths.extend(batch_paths)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = (all_preds == all_labels).mean()
        loss = running_loss / len(testloader)
        results=self.display_performance(all_preds,all_labels)
        
        return image_paths, all_preds, all_labels, accuracy, loss, results

    def display_performance(self, all_preds, all_labels, target_names=['real','fake']):
        print(f"✅ Debug: all_preds type: {type(all_preds)}, all_labels type: {type(all_labels)}")
        print(f"✅ Debug: all_preds shape: {len(all_preds)}, all_labels shape: {len(all_labels)}")

        print("\nClassification Report:")
        report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0, output_dict=True)
        print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(all_labels, all_preds, labels=[1, 0])
        print(cm)
        
        # Extract True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
        tn, fp, fn, tp = cm.ravel()
        results = {
            'precision': report['fake']['precision'],
            'recall': report['fake']['recall'],
            'f1-score': report['fake']['f1-score'],
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }
        
        return results

        
    def save_model(self, filename="model.pth"):
        """Saves the model weights in PyTorch format."""
        torch.save(self.model.state_dict(), os.path.join(self.output_model_dir, filename))
        print(f"Model saved at: {os.path.join(self.output_model_dir, filename)}")

    def save_training_history(self, history, filename="history.pkl"):
        """Saves training history to a pickle file."""
        with open(os.path.join(self.output_model_dir, filename), "wb") as f:
            pickle.dump(history, f)
        print(f"Training history saved")

        # Plot training history
        plt.figure(figsize=(8, 5))
        plt.plot(history["train_acc"], label="Train Accuracy")
        plt.plot(history["val_acc"], label="Val Accuracy")
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy/Loss")
        plt.legend()
        plt.grid()
        plt.title("Training History")
        plt.savefig(os.path.join(self.output_model_dir, "history_plot.png"))
        print(f"Training history plot saved at: {os.path.join(self.output_model_dir, 'history_plot.png')}")
