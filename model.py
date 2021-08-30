import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet as efficientNet
import timm

class BaseModel(nn.Module):
    def __init__(self, num_classes, *args):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


class ResNet(nn.Module):
    def __init__(self,
                 num_classes,
                 pretrained=False):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.selu = torch.nn.SELU()
        self.linear = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.selu(x)
        x = self.linear(x)
        return x


class VIT(nn.Module):
    # vit_base_patch16_224
    def __init__(self, model_name, num_classes, pretrained=False):
        super(VIT, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


class EfficientNet(nn.Module):
    # efficientnet-b0
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = efficientNet.from_pretrained(model_name, num_classes=num_classes)
        else:
            self.model = efficientNet.from_name(model_name, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
