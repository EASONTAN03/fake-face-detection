import os 
import numpy as np
import json
import csv
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer, log_loss, hinge_loss
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve, StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchsummary import summary
import matplotlib.pyplot as plt
from datetime import datetime

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
