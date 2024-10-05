import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18
from torchvision.models import resnet50

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)  # Binary output

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        feat = F.normalize(x, dim=1)
        return logits, feat

class ResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, model=None):
        super().__init__()
        if model == None:
            model = resnet18(pretrained=pretrained)
            for param in model.parameters():
                param.requires_grad = False

            modules = list(model.children())[:-1]
            self.extractor = nn.Sequential(*modules)
            self.embed_size = 512
            self.num_classes = num_classes
            self.fc = nn.Sequential(
               nn.Dropout(0.5),
               nn.Linear(model.fc.in_features, num_classes)
            )
            #self.fc = nn.Linear(self.embed_size, num_classes)
        else:
            modules = list(model.children())[:-1]
            self.extractor = nn.Sequential(*modules)
            self.embed_size = 512
            self.num_classes = num_classes
            self.fc = model.fc
        print(f"ResNet18 - num_classes: {num_classes} pretrained: {pretrained}")

    def forward(self, x):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)
        logits = self.fc(out)
        feat = F.normalize(out, dim=1)
        return logits, feat

class ResNet50(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, model=None):
        super().__init__()
        if model == None:
            model = resnet50(pretrained=pretrained)
            modules = list(model.children())[:-1]
            self.extractor = nn.Sequential(*modules)
            self.embed_size = 2048
            self.num_classes = num_classes
            self.fc = nn.Linear(self.embed_size, num_classes)
        else:
            modules = list(model.children())[:-1]
            self.extractor = nn.Sequential(*modules)
            self.embed_size = 2048
            self.num_classes = num_classes
            self.fc = model.fc
        print(f"ResNet50 - num_classes: {num_classes} pretrained: {pretrained}")

    def forward(self, x):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)
        logits = self.fc(out)
        feat = F.normalize(out, dim=1)
        return logits, feat
