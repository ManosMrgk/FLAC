import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121

class DenseNet121(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, model=None):
        super().__init__()
        if model is None:
            model = densenet121(pretrained=pretrained)

            self.extractor = nn.Sequential(*list(model.children())[:-1])

            self.embed_size = model.classifier.in_features
            self.num_classes = num_classes
            self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.embed_size, num_classes), 
                nn.Sigmoid()
            )
        else:
            self.extractor = nn.Sequential(*list(model.children())[:-1])
            self.embed_size = model.classifier.in_features  # DenseNet final layer input size
            self.num_classes = num_classes
            self.fc = model.classifier

        print(f"DenseNet121 - num_classes: {num_classes} pretrained: {pretrained}")

    def forward(self, x):
        features = self.extractor(x)
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1))

        out = pooled_features.view(pooled_features.size(0), -1)
        logits = self.fc(out)
        feat = F.normalize(out, dim=1)
        
        return logits, feat


