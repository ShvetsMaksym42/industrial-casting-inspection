import torch.nn as nn
import torch
import torchvision.models as models
from torchinfo import summary

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        return torch.relu(self.conv(x) + self.shortcut(x))

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.stage0 = nn.Sequential(
            ResidualBlock(16, 32),
            ResidualBlock(32,32)
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64,64)
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128,128)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
def get_model(model_name):
    if model_name == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1')
        #for param in model.parameters():
            #param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
        return model
    elif model_name == 'simple':
        return SimpleCNN()
    else:
        raise ValueError(
            f"Model '{model_name}' is not supported! "
            f"Available options are: ['simple', 'resnet18']"
        )

#model = SimpleCNN()
#summary(model, input_size=(16,3,512,512))