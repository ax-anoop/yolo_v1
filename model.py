import validator
import torch.nn as nn
import torch 
import torchvision
from torchsummary import summary
import utils 

'''
Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1
Output width = (Output width + padding width right + padding width left - kernel width) / (stride width) + 1
'''
# conv layer : channel_in, channel_out , kernel_size, stride, padding
# conv, output_channels, kernel, stride, padding
# maxp, None           , kernel, stride, padding 
maxP = ["maxp", 0, 2, 2, 0]
cov1 = [["conv", 256, 1, 1, 0], ["conv", 512, 3, 1, 1]]
cov2 = [["conv", 512, 1, 1, 0], ["conv", 1024, 3, 1, 1]]

darknet_arch = [
    ["conv", 64, 7, 2, 3], 
    maxP,
    ["conv", 192, 3, 1, 1],
    maxP,
    ["conv", 128, 1, 1, 0],
    ["conv", 256, 3, 1, 1],
    ["conv", 256, 1, 1, 0],
    ["conv", 512, 3, 1, 1],
    maxP, 
    cov1,cov1,cov1,cov1,
    ["conv", 512, 1, 1, 0],
    ["conv", 1024, 3, 1, 1],
    maxP,
    cov2, cov2
]

neck_arch = [
    ["conv", 1024, 3, 1, 1],
    ["conv", 1024, 3, 2, 1],
    ["conv", 1024, 3, 1, 1],
    ["conv", 1024, 3, 1, 1]
]

ml = []
for l in darknet_arch:
    if len(l)<5:
        for j in l:
            ml.append(j)
    else:
        ml.append(l)
darknet_arch = ml

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20, img_size=448, backbone="darknet"):
        super(Yolov1, self).__init__()
        self.darknet_arch = darknet_arch
        self.in_ch = in_channels
        self.img_size = img_size
        self.flatten = nn.Flatten()
        if backbone == "darknet":
            self.backbone = self._create_from_arch(darknet_arch, 3)
        elif backbone == "resnet152":
            # ResNet152_Weights.IMAGENET1K_V2
            self.backbone = nn.Sequential(*list(torchvision.models.resnet152(weights="IMAGENET1K_V2").children())[:-2])
        elif backbone == "resnet50": 
            self.backbone = nn.Sequential(*list(torchvision.models.resnet50(weights="IMAGENET1K_V1").children())[:-2])
        elif backbone == "resnet34": 
            self.backbone = nn.Sequential(*list(torchvision.models.resnet34(weights="IMAGENET1K_V1").children())[:-2])
        elif backbone == "vgg16": 
            self.backbone = nn.Sequential(*list(torchvision.models.resnet34(weights="IMAGENET1K_V1").children())[:-2])
        else:
            print("invalid backbone")
        print("backbone:", backbone)
        bb_output = self.backbone(torch.randn([1,3,img_size,img_size])).shape
        self.neck = self._create_from_arch(neck_arch, bb_output[1])
        self.head = self._create_fcs(split_size, num_boxes, num_classes)

    def forward(self, x):
        out = self.backbone(x)
        out = self.neck(out)
        out = self.flatten(out)
        out = self.head(out)
        return out

    def _create_fcs(self, split_size, num_boxes, num_classes, O=4096):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Linear(1024*S*S, O), #og paper, should be 4096
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(O, S * S * (C+B*5)),
        )

    # layers.append(nn.Flatten())
    def _create_from_arch(self, arch, in_ch):
        layers = []
        for l in arch:
            if l[0] == "conv":
                layer = CNNBlock(in_ch, l[1], kernel_size=l[2], stride=l[3], padding=l[4])
                in_ch = l[1]
            elif l[0] == "maxp":
                layer = nn.MaxPool2d(l[2], stride=l[3], padding=l[4])
            layers.append(layer)
        seq_layers = nn.Sequential(*layers)
        return seq_layers

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

def create(backbone="darknet"):
    return Yolov1(backbone, backbone=backbone)

if __name__ == '__main__':
    # v = validator.Val(model_layout)
    # v.check_model()
    model = Yolov1(3, backbone="resnet34")
    out = model(torch.randn(1,3,448,448))
    print(out.shape)

'''
1. Sales & marketing 
2. Grant application > Poketed, https://app.hellopocketed.io/
'''