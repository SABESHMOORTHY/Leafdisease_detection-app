# -------------------- IMPORTS --------------------
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
import pickle
import torch.nn.functional as F

# -------------------- CONFIG --------------------
dataset_path = "Datasetplant2"
batch_size = 32
epochs = 20  # Increased epochs
model_save_path = "resnet50_finetuned_10classes.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- TRANSFORMS --------------------
# Add normalization as per ResNet pretrained weights mean and std
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# -------------------- DATA LOADING --------------------
full_dataset = ImageFolder(root=dataset_path, transform=train_transform)
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Total images: {len(full_dataset)}")
print(f"Classes: {class_names}")

# -------------------- DATA SPLITTING --------------------
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Override validation dataset transform (no augmentation)
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -------------------- MODEL SETUP --------------------
model = models.resnet50(pretrained=True)

# Freeze all layers initially
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4 and fc for fine-tuning
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, num_classes)  # new classification layer
model = model.to(device)

# -------------------- LOSS AND OPTIMIZER --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

# Learning rate scheduler - reduce LR by 0.1 every 7 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# -------------------- TRAINING --------------------
best_val_acc = 0
early_stop_counter = 0
early_stop_patience = 5  # stop if val accuracy doesn't improve for 5 epochs

print("Training started...\n")
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    # Validation phase
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
          f"|| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Step learning rate scheduler
    scheduler.step()

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({'model_state_dict': model.state_dict()}, model_save_path)
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

# -------------------- LOAD BEST MODEL --------------------
model.load_state_dict(torch.load(model_save_path)["model_state_dict"])

# -------------------- SAVE MODEL AS PICKLE --------------------
model_for_pickle = models.resnet50(pretrained=False)
model_for_pickle.fc = nn.Linear(model_for_pickle.fc.in_features, num_classes)
model_for_pickle.load_state_dict(torch.load(model_save_path)["model_state_dict"])
model_for_pickle.eval()

model_bundle = {
    "model": model_for_pickle,
    "class_names": class_names
}

pickle_file_path = "resnet50_leaf_model.pkl"
with open(pickle_file_path, "wb") as f:
    pickle.dump(model_bundle, f)

print(f"\n Model also saved as Pickle to: {pickle_file_path}")
