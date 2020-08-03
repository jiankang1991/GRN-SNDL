

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
from torchvision import models


def weights_init_kaiming(m):
    # https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch/blob/master/main.py

    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data)

def fc_init_weights(m):
    # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch/49433937#49433937
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)





class ResNet18(nn.Module):
    def __init__(self, dim = 128):
        super().__init__()

        resnet = models.resnet18(pretrained=True)
        
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(512, dim)
        self.FC.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)

        x = self.FC(x)

        e = F.normalize(x)

        return e

class SiameseResNet18(nn.Module):
    def __init__(self, dim = 128):
        super().__init__()

        resnet = models.resnet18(pretrained=True)
        
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(512, dim)
        self.FC.apply(fc_init_weights)

    def forward(self, x1, x2):

        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        # print(x.shape)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.FC(x1)

        x2 = x2.view(x2.size(0), -1)
        x2 = self.FC(x2)

        return x1, x2



class ResNet18_cls(nn.Module):
    def __init__(self, clsNum, dim = 128):
        super().__init__()

        resnet = models.resnet18(pretrained=True)
        
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC1 = nn.Linear(512, dim)
        self.FC2 = nn.Linear(dim, clsNum)

        self.FC1.apply(fc_init_weights)
        self.FC2.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)

        x = self.FC1(x)
    
        e = F.normalize(x)

        logits = self.FC2(x)

        return e, logits





class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)



class InceptionV3(nn.Module):
    def __init__(self, dim = 128):
        super().__init__()

        inception = models.inception_v3(pretrained=True)

        self.encoder1 = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e
        )

        # self.AuxLogits = InceptionAux(768, numCls)

        self.encoder2 = nn.Sequential(
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1,1))
        )
        # self.dropout = nn.Dropout(0.2)
        # self.FC = nn.Linear(2048, numCls)
        self.FC = nn.Linear(2048, dim)
        self.FC.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder1(x)
        # aux = self.AuxLogits(x)
        x = self.encoder2(x)
        x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        x = self.FC(x)
        
        e = F.normalize(x)

        return e


class SiameseInceptionV3(nn.Module):
    def __init__(self, dim = 128):
        super().__init__()

        inception = models.inception_v3(pretrained=True)

        self.encoder1 = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e
        )

        # self.AuxLogits = InceptionAux(768, numCls)

        self.encoder2 = nn.Sequential(
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1,1))
        )
        # self.dropout = nn.Dropout(0.2)
        # self.FC = nn.Linear(2048, numCls)
        self.FC = nn.Linear(2048, dim)
        self.FC.apply(fc_init_weights)

    def forward(self, x1, x2):
        x1 = self.encoder1(x1)
        x2 = self.encoder1(x2)
        # aux = self.AuxLogits(x)
        x1 = self.encoder2(x1)
        x2 = self.encoder2(x2)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        # x = self.dropout(x)
        x1 = self.FC(x1)
        x2 = self.FC(x2)

        return x1, x2


class InceptionV3_cls(nn.Module):
    def __init__(self, clsNum, dim = 128):
        super().__init__()

        inception = models.inception_v3(pretrained=True)

        self.encoder1 = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e
        )

        # self.AuxLogits = InceptionAux(768, numCls)

        self.encoder2 = nn.Sequential(
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1,1))
        )
        # self.dropout = nn.Dropout(0.2)
        # self.FC = nn.Linear(2048, numCls)
        self.FC1 = nn.Linear(2048, dim)
        self.FC2 = nn.Linear(dim, clsNum)

        self.FC1.apply(fc_init_weights)
        self.FC2.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder1(x)
        # aux = self.AuxLogits(x)
        x = self.encoder2(x)
        x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        x = self.FC1(x)
    
        e = F.normalize(x)

        logits = self.FC2(x)

        return e, logits




class ResNet50(nn.Module):
    def __init__(self, dim = 128):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(2048, dim)
        self.FC.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)

        x = self.FC(x)

        e = F.normalize(x)

        return e


class SiameseResNet50(nn.Module):
    def __init__(self, dim = 128):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(2048, dim)
        self.FC.apply(fc_init_weights)

    def forward(self, x1, x2):

        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        # print(x.shape)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.FC(x1)

        x2 = x2.view(x2.size(0), -1)
        x2 = self.FC(x2)

        return x1, x2


class ResNet50_cls(nn.Module):
    def __init__(self, clsNum, dim = 128):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC1 = nn.Linear(2048, dim)
        self.FC2 = nn.Linear(dim, clsNum)

        self.FC1.apply(fc_init_weights)
        self.FC2.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)

        x = self.FC1(x)
    
        e = F.normalize(x)

        logits = self.FC2(x)

        return e, logits


class WideResNet50_2(nn.Module):
    def __init__(self, dim = 128):
        super().__init__()

        resnet = models.wide_resnet50_2(pretrained=True)
        
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(2048, dim)
        self.FC.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)

        x = self.FC(x)

        e = F.normalize(x)

        return e



class SiameseWideResNet50_2(nn.Module):
    def __init__(self, dim = 128):
        super().__init__()

        resnet = models.wide_resnet50_2(pretrained=True)
        
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(2048, dim)
        self.FC.apply(fc_init_weights)

    def forward(self, x1, x2):

        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        # print(x.shape)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.FC(x1)

        x2 = x2.view(x2.size(0), -1)
        x2 = self.FC(x2)

        return x1, x2

class WideResNet50_2_cls(nn.Module):
    def __init__(self, clsNum, dim = 128):
        super().__init__()

        resnet = models.wide_resnet50_2(pretrained=True)
        
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC1 = nn.Linear(2048, dim)
        self.FC2 = nn.Linear(dim, clsNum)

        self.FC1.apply(fc_init_weights)
        self.FC2.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)

        x = self.FC1(x)
    
        e = F.normalize(x)

        logits = self.FC2(x)

        return e, logits

if __name__ == "__main__":
    
    inputs = torch.randn(16,3,256,256)

    net = WideResNet50_2()

    outputs = net(inputs)

    print(outputs.shape)



