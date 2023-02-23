import torchvision.models as models
import torch.nn as nn
import torch
from torch.nn.functional import interpolate


class Resnet_Skip_Residual(nn.Module):
    def __init__(self, n_class, freeze_encoder=True):
        self.n_class = n_class
        self.freeze_encoder = freeze_encoder
        super(Resnet_Skip_Residual, self).__init__()
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        if self.freeze_encoder:
            for param in resnet.parameters():
                param.requires_grad = False
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512 + 256, 256, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256 + 128, 128, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128 + 64, 64, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

        self.onexone = nn.Conv2d(in_channels=768, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.onexone3 = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=1, stride=1, padding=0)

    def forward(self, images):
        x0 = self.resnet[0](images)
        x1 = self.resnet[1](x0)
        x2 = self.resnet[2](x1)
        x3 = self.resnet[3](x2)
        x4 = self.resnet[4](x3)
        x5 = self.resnet[5](x4)
        x6 = self.resnet[6](x5)
        out = self.resnet[7](x6)

        y1 = self.bn1(self.relu(self.deconv1(out)))
        y1 = torch.cat([y1, x6], dim=1)
        y2 = self.bn2(self.relu(self.deconv2(y1)))
        y2 = torch.cat([y2, x5], dim=1)
        y3 = self.bn3(self.relu(self.deconv3(y2))) + \
             self.onexone(interpolate(y1, size=[56, 56], mode='bilinear', align_corners=True))
        y3 = torch.cat([y3, x4], dim=1)
        y4 = self.bn4(self.relu(self.deconv4(y3)))
        y4 = torch.cat([y4, x2], dim=1)
        y5 = self.bn5(self.relu(self.deconv5(y4))) + \
             self.onexone3(interpolate(y3, size=[224, 224], mode='bilinear', align_corners=True))
        score = self.classifier(y5)
        return score
