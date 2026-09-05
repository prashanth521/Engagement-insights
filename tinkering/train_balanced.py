import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Custom Focal Loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def get_class_weights(data_dir):
    """Calculate class weights based on inverse frequency."""
    class_counts = []
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    
    for class_name in class_names:
        count = len(list((data_dir / class_name).glob('*.*')))
        class_counts.append(count)
    
    # Calculate weights (inverse of frequency)
    weights = 1.0 / np.sqrt(np.array(class_counts))  # Using sqrt for less aggressive weighting
    weights = weights / weights.sum() * len(weights)  # Normalize
    
    return torch.FloatTensor(weights)

def create_data_loaders(data_dir, batch_size=32, img_size=224):
    """Create data loaders with augmentation and balanced sampling."""
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/Test transforms
    val_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=str(data_dir / 'train'),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=str(data_dir / 'val'),
        transform=val_transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=str(data_dir / 'test'),
        transform=val_transform
    )
    
    # Calculate class weights for weighted sampling
    class_weights = get_class_weights(data_dir / 'train')
    sample_weights = [0] * len(train_dataset)
    for idx, (_, label) in enumerate(train_dataset.samples):
        sample_weights[idx] = class_weights[label]
    
    # Create weighted sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, 
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes

def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = running_loss / total
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def main():
    # Configuration
    data_dir = Path("C:/tinkering/tinkering/data/balanced_engagement_data")
    num_epochs = 50
    batch_size = 64
    learning_rate = 0.001
    num_classes = 4
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_dir, batch_size=batch_size
    )
    print(f"Classes: {class_names}")
    
    # Initialize model
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # Loss function with class weights
    class_weights = get_class_weights(data_dir / 'train').to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # Optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=5, verbose=True
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Training loop
    best_val_acc = 0.0
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"runs/engagement_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(output_dir))
    
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names
            }, output_dir / 'best_model.pth')
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping if needed (optional)
        if epoch > 10 and val_acc < 0.1:
            print("Early stopping due to poor performance")
            break
    
    # Test the model
    print("\nTesting...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'class_names': class_names
    }, output_dir / 'final_model.pth')
    
    print(f"\nTraining complete. Model saved to {output_dir}")

if __name__ == "__main__":
    import torchvision.models as models
    main()
