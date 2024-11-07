'''Borrowed and modified from 
@article{orhan2021semantic,
  title={Semantic segmentation of outdoor panoramic images},
  author={Orhan, Semih and Bastanlar, Yalin},
  journal={Signal, Image and Video Processing},
  year={2021},
  publisher={Springer}
  https://github.com/semihorhan/semseg-outdoor-pano.git
}
'''
import torch
import torch.nn as nn
from equiconv import DeformConv2d_plus_Offset


def convrelu(in_channels, out_channels, kernel, padding, layerdict, offsetdict, layer_number):
    return nn.Sequential(
        DeformConv2d_plus_Offset(in_channels, out_channels, kernel, padding=padding,
        offset_input=offsetdict[layerdict[layer_number]]),
        nn.ReLU(inplace=True),
    )


def equiconvrelu(in_channels, out_channels, kernel, padding, layerdict, offsetdict, layer_number):
    return nn.Sequential(
        DeformConv2d_plus_Offset(in_channels, out_channels, kernel, padding=padding,
        offset_input=offsetdict[layerdict[layer_number]]),
        nn.ReLU(inplace=True),
    )


def base_layer_0_3(layerdict, offsetdict):
    return nn.Sequential(
        DeformConv2d_plus_Offset(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False,
        offset_input=offsetdict[layerdict[2]]),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
    )


def base_layer_3_5(layerdict, offsetdict):
    return nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
        DeformConv2d_plus_Offset(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[3]]),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        DeformConv2d_plus_Offset(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[4]]),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

        DeformConv2d_plus_Offset(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[5]]),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        DeformConv2d_plus_Offset(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[6]]),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    )


def base_layer_5(layerdict, offsetdict):
    return nn.Sequential(
        DeformConv2d_plus_Offset(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[7]]),
        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        DeformConv2d_plus_Offset(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[8]]),
        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        DeformConv2d_plus_Offset(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[9]]),
        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        DeformConv2d_plus_Offset(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[10]]),
        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
   )


def base_layer_6(layerdict, offsetdict):
    return nn.Sequential(
        DeformConv2d_plus_Offset(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[11]]),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        DeformConv2d_plus_Offset(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[12]]),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        DeformConv2d_plus_Offset(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[13]]),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        DeformConv2d_plus_Offset(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[14]]),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    )


def base_layer_7(layerdict, offsetdict):
    return nn.Sequential(
        DeformConv2d_plus_Offset(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[15]]),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        DeformConv2d_plus_Offset(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[16]]),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        DeformConv2d_plus_Offset(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[17]]),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        DeformConv2d_plus_Offset(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        offset_input=offsetdict[layerdict[18]]),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    )



class PanoramaVisionEncoder(nn.Module):
    """Encoder part of the UNetEquiconv model."""
    def __init__(self, layer_dict, offset_dict):
        super(PanoramaVisionEncoder, self).__init__()

        self.layerdict = layer_dict
        self.offsetdict = offset_dict

        # Define only the encoder layers
        self.layer0 = base_layer_0_3(self.layerdict, self.offsetdict)
        self.layer1 = base_layer_3_5(self.layerdict, self.offsetdict)
        self.layer2 = base_layer_5(self.layerdict, self.offsetdict)
        self.layer3 = base_layer_6(self.layerdict, self.offsetdict)
        self.layer4 = base_layer_7(self.layerdict, self.offsetdict)
        self.layer4_1x1 = convrelu(512, 512, 1, 0, self.layerdict, self.offsetdict, layer_number=19)
        self.layer5 = nn.Sequential(
                    nn.Conv2d(512, 2048, 1, padding=0),
                    nn.ReLU(inplace=True),
                    )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer4 = self.layer4_1x1(layer4)
        layer5 = self.layer5(layer4)  # Shape: (B, 2048, H/32, W/32)
        layer5 = self.global_avg_pool(layer5)
        return layer5  # Final output from the encoder

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layerdict, offsetdict = torch.load('/home/erzurumlu.1/yunus/git/whereami/checkpoints/panorama_unet/layer_256x512.pt'), torch.load('/home/erzurumlu.1/yunus/git/whereami/checkpoints/panorama_unet/offset_256x512.pt')

    model = PanoramaVisionEncoder(layer_dict=layerdict, offset_dict=offsetdict)
    model = model.to(device)
    input0 = torch.randn(2, 3, 256, 512)
    input0 = input0.to(device)
    output0 = model(input0)
    print(f"Output shape: {output0.shape}") # 2, 512, 8, 16
