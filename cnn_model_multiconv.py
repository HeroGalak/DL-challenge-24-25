import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # set seed for reproducibility

# path where best model is saved
model_save_path = "/home/francesco/Downloads/best_model_convnet.pth"
best_valid_accuracy = 0.0

# Data augmentation and conversion to tensor
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

# Dataset path
train_dataset = datasets.ImageFolder(root="/home/francesco/Downloads/dl2425_challenge_dataset/train", transform=transform)
valid_dataset = datasets.ImageFolder(root="/home/francesco/Downloads/dl2425_challenge_dataset/val", transform=transform)

# Load the dataset
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=8)

# Model definition
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) 
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256 * 14 * 14, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x) # drop a neuron with probability 0.5
        x = torch.sigmoid(self.fc(x))  # Output a probability
        return x

# Execute training only if the file is directly executed
if __name__ == "__main__":
    model = MultiConvNetWithBN()
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3) # Adam optimizer, starting LR, weight decay
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) # halve LR after 3 epochs without improvement
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    num_epochs = 200 # max number of epochs
    epochs_patience = 15  # max number of epochs without improvement (early stopping)
    epochs_no_improve = 0  # counter of epochs with no improvement
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device, dtype=torch.float32)
            
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            
            # backword pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # compute training accuracy
            predicted = (outputs > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels.unsqueeze(1)).sum().item()
        
        train_accuracy = 100 * correct_train / total_train
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")
        
        # validation
        model.eval()
        valid_loss = 0.0
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device, dtype=torch.float32)
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1))
                valid_loss += loss.item()
                
                # compute validation accuracy
                predicted = (outputs > 0.5).float()
                total_valid += labels.size(0)
                correct_valid += (predicted == labels.unsqueeze(1)).sum().item()
        
        valid_accuracy = 100 * correct_valid / total_valid
        print(f"Validation Loss: {valid_loss/len(valid_loader):.4f}, Validation Accuracy: {valid_accuracy:.2f}%")
        scheduler.step(valid_loss / len(valid_loader))

        # check accuracy improvement for early stopping
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"Modello salvato con accuracy: {best_valid_accuracy:.2f}%")
            epochs_no_improve = 0  # Reset if a better accuracy is reached
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation accuracy for {epochs_no_improve} epochs.")
        
        # check for early stopping
        if epochs_no_improve >= epochs_patience:
            print("Early stopping triggered!")
            break
