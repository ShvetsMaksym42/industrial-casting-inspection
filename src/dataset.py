import os
import cv2
import torch
from torch.utils.data import Dataset

class DefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.class_map = {
            "def_front": 1,
            "ok_front": 0
            }

        for class_name in self.class_map:
            class_path = os.path.join(root_dir, class_name)
            for img_name in sorted(os.listdir(class_path)):
                img_path = os.path.join(class_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.class_map[class_name])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return image, label