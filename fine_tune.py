import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os

# Define model architecture (same as in app.py)
class SignLanguageModel(nn.Module):
    def __init__(self, num_classes=26, pretrained=False):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def main():
    dataset_path = 'custom_dataset'
    
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print(f"Error: Could not find any data in '{dataset_path}'.")
        print("Please run collect_data.py first to gather some images.")
        return

    print("Setting up dataset and model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We must ensure the classes map exactly to 0-25 for A-Z
    # so that we don't mess up the original model's weights.
    # ImageFolder assigns indices alphabetically by folder name.
    # We will enforce class_to_idx mapping.
    
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, root, transform=None):
            self.root = root
            self.transform = transform
            self.samples = []
            
            for folder_name in os.listdir(root):
                folder_path = os.path.join(root, folder_name)
                if os.path.isdir(folder_path) and len(folder_name) == 1 and 'A' <= folder_name <= 'Z':
                    class_idx = ord(folder_name) - 65
                    for file_name in os.listdir(folder_path):
                        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.samples.append((os.path.join(folder_path, file_name), class_idx))
                            
        def __len__(self):
            return len(self.samples)
            
        def __getitem__(self, idx):
            path, target = self.samples[idx]
            from PIL import Image
            img = Image.open(path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img, target

    # Preprocessing transform (matching training & inference)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), # Add slight augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Slight color change
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    try:
        train_dataset = CustomDataset(dataset_path, transform=train_transforms)
        # Filter out classes with 0 images so DataLoader doesn't complain, 
        # but actually PyTorch handles empty classes fine during iteration as long as they just don't appear in batches.
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if len(train_dataset) == 0:
        print("No valid images found in custom_dataset folders.")
        return

    batch_size = min(32, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"Found {len(train_dataset)} images across your custom dataset.")

    # Load existing model
    model = SignLanguageModel(num_classes=26)
    model_path = 'best_model.pth'
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded 'best_model.pth' as the starting point.")
    except Exception as e:
        print(f"Error loading existing model: {e}")
        return

    model.to(device)
    
    # Fine-tuning setup
    # We use a very small learning rate because the model is already trained.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 5
    print("\nStarting Fine-Tuning Process...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy on custom data: {epoch_acc:.2f}%")
        
    print("\nFinished Fine-Tuning!")
    
    # Save the new model
    save_path = 'fine_tuned_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_acc': epoch_acc # just saving the training acc as a placeholder
    }, save_path)
    
    print(f"Saved new model to '{save_path}'")
    print("You can now update 'app.py' to use 'fine_tuned_model.pth' instead of 'best_model.pth'!")

if __name__ == "__main__":
    main()
