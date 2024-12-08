import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

model_save_path = "/home/francesco/Downloads/best_model_resnet.pth"

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root="/home/francesco/Downloads/dl2425_challenge_dataset/train", transform=transform)
valid_dataset = datasets.ImageFolder(root="/home/francesco/Downloads/dl2425_challenge_dataset/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=6)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True) # load pretrained resnet18

# modify the last layer for binary classification
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 1),
    nn.Sigmoid() # to output a probability
)
model = model.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

num_epochs = 50
best_valid_accuracy = 0.0
epochs_patience = 10
epochs_no_improve = 0

# training
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device, dtype=torch.float32)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels.unsqueeze(1)).sum().item()

    train_accuracy = 100 * correct_train / total_train
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # validation
    model.eval()
    valid_loss = 0.0
    correct_valid = 0
    total_valid = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.float32)

            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))

            valid_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total_valid += labels.size(0)
            correct_valid += (predicted == labels.unsqueeze(1)).sum().item()

    valid_accuracy = 100 * correct_valid / total_valid
    print(f"Validation Loss: {valid_loss/len(valid_loader):.4f}, Validation Accuracy: {valid_accuracy:.2f}%")
    scheduler.step(valid_loss / len(valid_loader))

    # save best model
    if valid_accuracy > best_valid_accuracy:
        best_valid_accuracy = valid_accuracy
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved with accuracy: {best_valid_accuracy:.2f}%")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"No improvement in validation accuracy for {epochs_no_improve} epochs.")

    # early stopping
    if epochs_no_improve >= epochs_patience:
        print("Early stopping triggered!")
        break

print("Training completed!")
