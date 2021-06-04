import torchvision.models
from torch import nn


def resnet18(num_classes=10, *args, **kwargs):
    model = torchvision.models.resnet18(num_classes=10, *args, **kwargs)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    return model
