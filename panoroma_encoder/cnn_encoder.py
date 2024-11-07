
import torch
import torch.nn as nn
from torchvision import models


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class PanoramaVisionEncoder(nn.Module):
    def __init__(self):
        super(PanoramaVisionEncoder, self).__init__()
        self.base_model = models.resnet18(pretrained=False)
        self.base_layers = list(self.base_model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]
        self.layer4_1x1 = convrelu(512, 512, 1, 0)


        self.layer5 = convrelu(512, 2048, 1, 0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer4 = self.layer4_1x1(layer4)
        layer5 = self.layer5(layer4)  # Shape: (B, 2048, H/32, W/32)
        layer5 = self.global_avg_pool(layer5)

        return layer5

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PanoramaVisionEncoder()
    model = model.to(device)
    input0 = torch.randn(2, 3, 256, 512)
    input0 = input0.to(device)
    output0 = model(input0)
    print(f"Output shape: {output0.shape}") # this returns 2, 512, 8, 16
