import os 
import numpy as np
import json
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer, log_loss
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import random
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

class ModelTrainer:
    def __init__(self, model="svm", checkpoint_dir='checkpoints', model_benchmark=1, random_state=42):
        
        self.random_state = random_state
        self.benchmark = model_benchmark
        self.scaler = StandardScaler()
        if model=="svm":
            self.svm_model = None
            self.best_svm = None
            self.training_history = {'svm': {'iterations': 0, 'best_score': float('inf')}}
        if model=="knn":
            self.knn_model = None
            self.best_knn = None
            self.training_history = {'svm': {'iterations': 0, 'best_score': float('inf')}}
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
    def _save_checkpoint(self, model_type):
        """
        Save model checkpoint and training history
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the model
        if model_type == 'svm':
            model = self.best_svm
            history = self.training_history['svm']
        else:
            model = self.best_knn
            history = self.training_history['knn']
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'{model_type}_checkpoint_{timestamp}_{self.benchmark}.joblib'
        )
        history_path = os.path.join(
            self.checkpoint_dir,
            f'{model_type}_history_{timestamp}_{self.benchmark}.json'
        )
        
        # Save model
        joblib.dump(model, checkpoint_path)
        
        # Save training history
        with open(history_path, 'w') as f:
            json.dump(history, f)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path, history_path
    
    def load_checkpoint(self, model_path, history_path=None):
        """
        Load model and training history from checkpoint
        """
        # Load model
        loaded_model = joblib.load(model_path)
        
        # Determine model type from filename
        if 'svm' in model_path.lower():
            self.best_svm = loaded_model
            model_type = 'svm'
        elif 'knn' in model_path.lower():
            self.best_knn = loaded_model
            model_type = 'knn'
        else:
            raise ValueError("Unknown model type in checkpoint file")
        
        # Load history if provided
        if history_path and os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history[model_type] = json.load(f)
        
        print(f"Checkpoint loaded: {model_path}")
        return loaded_model
    
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
    
    
    def train_svm(self, X_train, y_train, param_grid=None, continue_training=False):
        """
        Train SVM model with hyperparameter tuning
        """
        # Load latest checkpoint if continuing training
        if continue_training:
            print("\nContinuing training from checkpoints...")
            model_path, history_path = self.get_latest_checkpoint('svm')
            if model_path:
                self.load_checkpoint(model_path, history_path)
                print("Continuing SVM training from checkpoint...")
        
        # Define SVM pipeline
        svm_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True))
        ])
        
        if param_grid is None:
            param_grid = {'svm__C': [0.1, 1, 10, 100], 'svm__kernel': ['rbf', 'linear'], 'svm__gamma': ['scale', 'auto', 0.1, 0.01]}
        
        # Create custom scorer that includes loss
        scorer = make_scorer(log_loss, needs_proba=True, greater_is_better=False)
        
        # Perform grid search with cross-validation
        self.svm_model = GridSearchCV(
            svm_pipeline,
            param_grid,
            cv=5,
            scoring=scorer,
            n_jobs=-1,
            verbose=1
        )
        
        self.svm_model.fit(X_train, y_train)
        self.best_svm = self.svm_model.best_estimator_
        
        # Update training history
        self.training_history['svm']['iterations'] += 1
        current_score = -self.svm_model.best_score_  # Convert to positive loss
        if current_score < self.training_history['svm']['best_score']:
            self.training_history['svm']['best_score'] = current_score
            # Save checkpoint when we get a better score
            self._save_checkpoint('svm')
        
        return self.svm_model
    
    def train_knn(self, X_train, y_train, param_grid=None, continue_training=False):
        """
        Train KNN model with hyperparameter tuning
        """
        # Load latest checkpoint if continuing training
        if continue_training:
            model_path, history_path = self.get_latest_checkpoint('knn')
            if model_path:
                self.load_checkpoint(model_path, history_path)
                print("Continuing KNN training from checkpoint...")
        
        # Define KNN pipeline
        knn_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ])
        
        # Define parameter grid
        if param_grid is None:
           param_grid = {'knn__n_neighbors': [3, 5, 7, 9, 11],'knn__weights': ['uniform', 'distance'],'knn__metric': ['euclidean', 'manhattan']}
        
        # Create custom scorer that includes loss
        scorer = make_scorer(log_loss, needs_proba=True, greater_is_better=False)
        
        # Perform grid search with cross-validation
        self.knn_model = GridSearchCV(
            knn_pipeline,
            param_grid,
            cv=5,
            scoring=scorer,
            n_jobs=-1,
            verbose=1
        )
        
        self.knn_model.fit(X_train, y_train)
        self.best_knn = self.knn_model.best_estimator_
        
        # Update training history
        self.training_history['knn']['iterations'] += 1
        current_score = -self.knn_model.best_score_  # Convert to positive loss
        if current_score < self.training_history['knn']['best_score']:
            self.training_history['knn']['best_score'] = current_score
            # Save checkpoint when we get a better score
            self._save_checkpoint('knn')
        
        return self.knn_model
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate both models and compare their performance
        """
        results = {}
        
        # Evaluate SVM
        if self.best_svm is not None:
            svm_pred = self.best_svm.predict(X_test)
            svm_prob = self.best_svm.predict_proba(X_test)
            results['svm'] = {
                'accuracy': accuracy_score(y_test, svm_pred),
                'log_loss': log_loss(y_test, svm_prob),
                'classification_report': classification_report(y_test, svm_pred),
                'best_params': self.svm_model.best_params_,
                'training_iterations': self.training_history['svm']['iterations'],
                'best_score': self.training_history['svm']['best_score']
            }
            y_pred=svm_pred
        
        # Evaluate KNN
        if self.best_knn is not None:
            knn_pred = self.best_knn.predict(X_test)
            knn_prob = self.best_knn.predict_proba(X_test)
            results['knn'] = {
                'accuracy': accuracy_score(y_test, knn_pred),
                'log_loss': log_loss(y_test, knn_prob),
                'classification_report': classification_report(y_test, knn_pred),
                'best_params': self.knn_model.best_params_,
                'training_iterations': self.training_history['knn']['iterations'],
                'best_score': self.training_history['knn']['best_score']
            }
            y_pred=knn_pred
        
        return y_pred,results
    
    def plot_learning_curves(self, X, y, cv=5):
        """
        Plot learning curves for both models
        """
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        plt.figure(figsize=(12, 4))
        
        # Plot SVM learning curve if model exists
        if self.best_svm is not None:
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
        if self.best_knn is not None:
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