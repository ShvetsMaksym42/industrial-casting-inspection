import os
from src.dataset import DefectDataset
from src.model import get_model
from src.visualize import heatmap
import torch
import torch_directml
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import albumentations as A
from albumentations.pytorch import ToTensorV2

def main():

    MODEL_FILE_NAME = 'best_simple_model'
    MODEL_TYPE = 'simple' #or 'resnet18'

    THRESHOLD = 0.5

    ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(ROOT, "data")
    MODEL_PATH = os.path.join(ROOT, f"weights/{MODEL_FILE_NAME}.pth")

    device = torch_directml.device() if torch_directml.is_available() else 'cpu'
    model = get_model(MODEL_TYPE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    VISUALIZATION_TARGET_LAYER = model.layer4[-1] if MODEL_TYPE=='resnet18' else model.stage2[-1].conv[-2]

    test_transform = A.Compose([
        A.LongestMaxSize(max_size=512),
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=0),
        A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    full_dataset = DefectDataset(root_dir=DATA_PATH, transform=test_transform)
    labels = full_dataset.labels

    _, test_indices = train_test_split(
    list(range(len(labels))),
    test_size=0.2,
    stratify = labels,
    random_state=42
    )
    test_dataset= Subset(full_dataset, test_indices)
    test_loader= DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image= image.to(device)
            output = model(image).flatten()
            prob = torch.sigmoid(output).item()
            pred = 1 if prob>THRESHOLD else 0
            actual = label.item()

            all_preds.append(pred)
            all_labels.append(actual)
            if pred != actual:  #pred != actual  or  i%20==0
                err_type = 'FN' if actual==1 else 'FP'
                save_path = os.path.join(ROOT, f'error_visualization/model_{MODEL_TYPE}_error_{i}_{err_type}.jpg')
                heatmap(image, save_path, model, VISUALIZATION_TARGET_LAYER)

    print(f"\n--- Results for {MODEL_TYPE} ---")
    print(confusion_matrix(all_labels, all_preds))
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Defect']))

if __name__ == '__main__':
    main()