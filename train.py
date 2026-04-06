import os
from src.dataset import DefectDataset
from src.model import get_model
import torch
import torch_directml
import torchvision
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

def main():

    MODEL_TYPE = 'simple' # 'resnet18'  or  'simple'

    LR = 1e-5 if MODEL_TYPE=='resnet18' else 1e-4 #5e-4
    ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(ROOT, "data")
    LOG_PATH  = os.path.join(ROOT, "logs", f"defect_detection_{MODEL_TYPE}_run_1")
    SAVE_PATH = os.path.join(ROOT, "weights")
    os.makedirs(SAVE_PATH, exist_ok=True)

    device = torch_directml.device() if torch_directml.is_available() else 'cpu'
    model = get_model(MODEL_TYPE).to(device)

    writer = SummaryWriter(LOG_PATH)

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomResizedCrop(size=(512, 512), scale=(0.85, 1.0), ratio=(0.95, 1.05), p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.5),
        A.OneOf([
            A.GaussNoise(std_range=(0.04, 0.2), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.4),
        A.CoarseDropout(
            num_holes_range=(8, 15), 
            hole_height_range=(8, 12), 
            hole_width_range=(8, 12), 
            fill_value=0, 
            p=0.5),
        A.LongestMaxSize(max_size=512),
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=0),
        A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    test_transform = A.Compose([
        A.LongestMaxSize(max_size=512),
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=0),
        A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_dataset = DefectDataset(root_dir=DATA_PATH, transform=train_transform)
    test_dataset = DefectDataset(root_dir=DATA_PATH, transform=test_transform)
    labels = train_dataset.labels
    train_indices, test_indices = train_test_split(
    list(range(len(labels))),
    test_size=0.2,
    stratify = labels,
    random_state=42
    )
    train_dataset= Subset(train_dataset, train_indices)
    test_dataset= Subset(test_dataset, test_indices)
    train_loader= DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=3)
    test_loader= DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=3)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.6])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    images, labels = next(iter(train_loader))
    img_grid = torchvision.utils.make_grid(images[:8], normalize=True)
    writer.add_image('Augmented_Samples_Preview', img_grid)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.1) #not for resnet
    epochs = 125 #50 for resnet
    best_val_loss = 2.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.flatten(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).flatten()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_val_loss = val_loss / len(test_loader)
        accuracy = 100 * correct / total
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/test', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/test', accuracy, epoch)
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {accuracy:.2f}%")
        print("-" * 30)
        if  avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_name = os.path.join(SAVE_PATH, f'best_{MODEL_TYPE}_model.pth')
            torch.save(model.state_dict(), model_name)
        scheduler.step()

    model_name = os.path.join(SAVE_PATH, f'final_{MODEL_TYPE}_model.pth')
    torch.save(model.state_dict(), model_name)
    writer.close()

if __name__ == '__main__':
    main()