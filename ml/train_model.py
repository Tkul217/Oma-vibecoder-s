"""
Training Script for Car Condition Analysis Models

This script demonstrates how to train models for:
1. Cleanliness detection (clean/dirty)
2. Damage assessment (intact/damaged)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np

class CarDataset(Dataset):
    """Custom dataset for car condition analysis."""
    
    def __init__(self, data_dir, annotations_file, transform=None, task='cleanliness'):
        """
        Args:
            data_dir: Directory with car images
            annotations_file: JSON file with image annotations
            transform: Optional transform to be applied on images
            task: 'cleanliness' or 'condition'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.task = task
        
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_info = self.annotations[idx]
        img_path = os.path.join(self.data_dir, img_info['filename'])
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get label based on task
        if self.task == 'cleanliness':
            label = 0 if img_info['cleanliness'] == 'clean' else 1
        else:  # condition
            label = 0 if img_info['condition'] == 'intact' else 1
        
        return image, label

class CarConditionModel(nn.Module):
    """Model for binary classification of car conditions."""
    
    def __init__(self, num_classes=2):
        super(CarConditionModel, self).__init__()
        
        # Use pre-trained ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=True)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def get_transforms():
    """Get data transforms for training and validation."""
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_model(model, train_loader, val_loader, num_epochs=20, device='cuda'):
    """Train the model."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step()
        print()
    
    return model

def create_sample_annotations():
    """Create sample annotations file for demonstration."""
    
    # Sample data structure
    sample_annotations = [
        {
            "filename": "car_001.jpg",
            "cleanliness": "clean",
            "condition": "intact"
        },
        {
            "filename": "car_002.jpg",
            "cleanliness": "dirty",
            "condition": "intact"
        },
        {
            "filename": "car_003.jpg",
            "cleanliness": "clean",
            "condition": "damaged"
        },
        {
            "filename": "car_004.jpg",
            "cleanliness": "dirty",
            "condition": "damaged"
        }
    ]
    
    with open('sample_annotations.json', 'w') as f:
        json.dump(sample_annotations, f, indent=2)
    
    print("Sample annotations file created: sample_annotations.json")

def main():
    """Main training function."""
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample annotations (replace with your actual data)
    create_sample_annotations()
    
    # Data preparation (commented out since we don't have actual data)
    """
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = CarDataset('data/train', 'train_annotations.json', 
                              transform=train_transform, task='cleanliness')
    val_dataset = CarDataset('data/val', 'val_annotations.json', 
                            transform=val_transform, task='cleanliness')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Create and train cleanliness model
    cleanliness_model = CarConditionModel(num_classes=2).to(device)
    cleanliness_model = train_model(cleanliness_model, train_loader, val_loader, device=device)
    torch.save(cleanliness_model.state_dict(), 'cleanliness_model.pth')
    
    # Create and train condition model
    condition_dataset_train = CarDataset('data/train', 'train_annotations.json', 
                                       transform=train_transform, task='condition')
    condition_dataset_val = CarDataset('data/val', 'val_annotations.json', 
                                     transform=val_transform, task='condition')
    
    condition_train_loader = DataLoader(condition_dataset_train, batch_size=32, shuffle=True, num_workers=4)
    condition_val_loader = DataLoader(condition_dataset_val, batch_size=32, shuffle=False, num_workers=4)
    
    condition_model = CarConditionModel(num_classes=2).to(device)
    condition_model = train_model(condition_model, condition_train_loader, condition_val_loader, device=device)
    torch.save(condition_model.state_dict(), 'condition_model.pth')
    """
    
    print("Training script prepared. Add your actual dataset to run training.")
    print("\nDataset structure should be:")
    print("data/")
    print("  train/")
    print("    car_001.jpg")
    print("    car_002.jpg")
    print("    ...")
    print("  val/")
    print("    car_101.jpg")
    print("    car_102.jpg")
    print("    ...")
    print("  train_annotations.json")
    print("  val_annotations.json")

if __name__ == "__main__":
    main()