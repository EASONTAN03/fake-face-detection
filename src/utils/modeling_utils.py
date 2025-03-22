import os 
import numpy as np
import json
import csv
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from fvcore.nn import FlopCountAnalysis
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer, log_loss, hinge_loss
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve, StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchsummary import summary
import matplotlib.pyplot as plt
import time

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import operation_utils as utils
from modeling_utils import *

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

class EfficientNet(BaseCNN):
    def __init__(self, num_classes):
        super(EfficientNet, self).__init__(num_classes)
        
        # Load EfficientNet-B0 model
        self.model = models.efficientnet_b0(pretrained=True)

        # Replace the final classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        # self.model.classifier = nn.Sequential(
        #     nn.Linear(in_features, 1280),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.4),  # 🔹 Increased dropout from default (0.2) to 0.4
        #     nn.Linear(1280, num_classes)
        # )


    def forward(self, x):
        return self.model(x)

class MobileNet(BaseCNN):
    def __init__(self, num_classes):
        super(MobileNet, self).__init__(num_classes)
        self.model = models.mobilenet_v3_large(pretrained=True)

        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, num_classes)

        in_features = self.model.classifier[0].in_features  # This is 960 in MobileNetV3 Large
        # self.model.classifier = nn.Sequential(
        #     nn.Linear(in_features, 1280),  # Correct input size from 960 to 1280
        #     nn.Hardswish(),
        #     nn.Dropout(p=0.4),  # Increased dropout to prevent overfitting
        #     nn.Linear(1280, num_classes)  # Output to match number of classes
        # )

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
            "EfficientNet": EfficientNet,
            "MobileNet": MobileNet,
        }

        model_name=  MODEL_MAPPING.get(self.config["model_name"])
        print(model_name)

        if model_name is None:
            raise ValueError(f"Invalid model name '{self.config['model_name']}' in config file.")
        # self.model = model_name(channel=self.config["input_shape"][2], num_classes=num_classes).to(self.device)
        self.model = model_name(num_classes=num_classes).to(self.device)

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
            self.scheduler = self.get_scheduler()  # ✅ Add LR Scheduler
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
            "adam": optim.Adam(self.model.parameters(), lr=self.config["learning_rate"], weight_decay=1e-4),
            # "adam": optim.Adam(self.model.parameters(), lr=self.config["learning_rate"]),
            "sgd": optim.SGD(self.model.parameters(), lr=self.config["learning_rate"], momentum=0.9, weight_decay=1e-4),
            # "rmsprop": optim.RMSprop(self.model.parameters(), lr=self.config["learning_rate"])
        }
        return optimizers.get(self.config["optimizer"], optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4))

    def get_scheduler(self):
        """Returns the Cosine Annealing Learning Rate Scheduler."""
        if "scheduler" in self.config and self.config["scheduler"]["type"] == "cosine_annealing":
            return lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config["scheduler"]["T_max"], 
                eta_min=self.config["scheduler"]["eta_min"]
            )
        return None  # No scheduler if not specified

    def get_loss_function(self):
        """Returns the loss function specified in the JSON config."""
        loss_functions = {
            "cross_entropy": nn.CrossEntropyLoss(),
            "mse": nn.MSELoss(),
            "bce_logit": nn.BCEWithLogitsLoss()
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
            wrong_train, correct_train = 0.0, 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images) #.squeeze(dim=1)  # Remove extra dimension from outputs
                # labels = labels.float()  # Reshape labels to match outputs
                loss = self.loss_function(outputs, labels)  # Compute BCE loss
                loss.backward()
                self.optimizer.step()
                wrong_train += loss.item() * images.size(0) 
                correct_train += (outputs.argmax(1) == labels).sum().item()

            wrong_val, correct_val = 0.0, 0
            self.model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.loss_function(outputs, labels)
                    wrong_val += loss.item() * images.size(0) 
                    correct_val += (outputs.argmax(1) == labels).sum().item()

            train_loss = wrong_train / len(train_loader.dataset)
            val_loss = wrong_val / len(val_loader.dataset)

            train_acc = correct_train / len(train_loader.dataset)
            val_acc = correct_val / len(val_loader.dataset)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
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
        
            if self.scheduler is not None:
                self.scheduler.step()

            # # Reduce LR on Plateau
            # if self.early_stopping_counter >= self.config["callbacks"]["reduce_lr"]["patience"]:
            #     print("lr patience reached")
            #     old_lr = self.optimizer.param_groups[0]['lr']
            #     self.optimizer.param_groups[0]['lr'] *= self.config["callbacks"]["reduce_lr"]["factor"]
            #     print(f"Learning rate reduced from {old_lr} to {self.optimizer.param_groups[0]['lr']}")

        return history
    
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
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'accuracy': report['accuracy'],
            'precision': report['fake']['precision'],
            'recall': report['fake']['recall'],
            'f1-score': report['fake']['f1-score']
        }
        
        return results

    def evaluate_model(self, testloader):
        self.model.eval()
        all_preds, all_labels, pred_probs, image_paths = [], [], [], []
        inference_times = []
        wrong_train=0.0

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(testloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                start_time = time.time()
                outputs = self.model(inputs)
                inference_times.append(time.time() - start_time)

                loss = self.loss_function(outputs, labels)
                wrong_train += loss.item() * inputs.size(0) 

                # Calculate probabilities
                probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probabilities for "fake" class
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                pred_probs.extend(probabilities.cpu().numpy())

                # Store image paths
                batch_paths = [testloader.dataset.imgs[i][0] for i in range(batch_idx * testloader.batch_size, batch_idx * testloader.batch_size + len(labels))]
                image_paths.extend(batch_paths)

        test_loss = wrong_train / len(testloader.dataset)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        pred_probs = np.array(pred_probs)

        results=self.display_performance(all_preds,all_labels)
        auc_roc = roc_auc_score(all_labels, pred_probs)
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to milliseconds

        # Calculate FLOPs
        dummy_input = torch.randn(1, self.config["input_shape"][2], self.config["input_shape"][0], self.config["input_shape"][1]).to(self.device)
        flops = FlopCountAnalysis(self.model, dummy_input)
        total_flops = flops.total() / 1e9  # GFLOPs

        # Add at the end:
        add_result = {
            'auc_roc': auc_roc,
            'loss': test_loss,
            'inference_time_ms': avg_inference_time,
            'flops_giga': total_flops,
        }

        results = {**results, **add_result}

        print("\nEvaluation Metrics:")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"Average Inference Time: {avg_inference_time:.2f} ms")
        print(f"Total FLOPs: {total_flops:.2f} GFLOPs")
        print(f"Test Loss: {test_loss:.4f}")

        self.save_aucroc(all_labels, pred_probs, auc_roc)

        return image_paths, all_preds, all_labels, results

    def save_model(self, filename="model.pth"):
        """Saves the model weights in PyTorch format."""
        torch.save(self.model.state_dict(), os.path.join(self.output_model_dir, filename))
        print(f"Model saved at: {os.path.join(self.output_model_dir, filename)}")

    def save_aucroc(self, all_labels, pred_probs, auc_roc):
        fpr, tpr, _ = roc_curve(all_labels, pred_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc_roc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        roc_curve_path = os.path.join(self.output_model_dir, "roc_curve.png")
        plt.savefig(roc_curve_path)
        plt.close()
        print(f"✅ ROC curve saved at: {roc_curve_path}")

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
        plt.close()
        print(f"Training history plot saved at: {os.path.join(self.output_model_dir, 'history_plot.png')}")
