import torch
import torch.nn as nn
import numpy as np
import math

__all__ = ['BNWideResNet', 'bn_wideresnet16']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, convShortcut=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.convShortcut = convShortcut

    def forward(self, x):

        out = self.bn1(x)
        relu1 = self.relu(out)

        out = self.conv1(relu1)
        out = self.bn2(out)         
        out = self.relu(out)

        out = self.conv2(out)
        

        if self.convShortcut is not None:
            out += self.convShortcut(relu1)
        else:
            out += x
        return out

    def forward_hl(self, x):

        out1 = self.bn1(x)
        relu1 = self.relu(out1)

        out = self.conv1(relu1)
        out2 = self.bn2(out)
        out = self.relu(out2)

        out = self.conv2(out)

        if self.convShortcut is not None:
            out += self.convShortcut(relu1)
        else:
            out += x
        return out, out1, out2

class NetworkBlock(nn.Module):
    def __init__(self, block, in_planes, out_planes, blocks, start_num, stride=1):
        super(NetworkBlock, self).__init__()
        assert(blocks >= 1)
        self.layers = self._make_layer(block, in_planes, out_planes, blocks, stride)
        self.start_num = start_num # starting layer of the residual block
        self.layer_nn = nn.Sequential(*self.layers)

    def _make_layer(self, block, in_planes, out_planes, blocks, stride):
        convShortcut = nn.Conv2d(
            in_planes, 
            out_planes, 
            kernel_size=1, 
            stride=stride, 
            padding=0, 
            bias=False)

        layers = []
        layers.append(block(in_planes, out_planes, stride, convShortcut))
        for _ in range(1, blocks):
            layers.append(block(out_planes, out_planes))

        return layers

    def forward(self, x):
        return self.layer_nn(x)

    # forward pass computing hidden layers
    def forward_hl(self, x):
        layer_dict = {}
        out, out1, out2 = self.layers[0].forward_hl(x)
        layer_dict["block_%d_res" % (self.start_num)] = out
        layer_dict["block_%d_conv1" % (self.start_num)] = out1
        layer_dict["block_%d_conv2" % (self.start_num)] = out2 
        for i in range(1, len(self.layers)):
            layer_ind = self.start_num + i
            out, out1, out2 = self.layers[i].forward_hl(out)
            layer_dict["block_%d_res" % (layer_ind)] = out 
            layer_dict["block_%d_conv1" % (layer_ind)] = out1 
            layer_dict["block_%d_conv2" % (layer_ind)] = out2 
        return out, layer_dict

class BNWideResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, width=10):
        super(BNWideResNet, self).__init__()
        self.num_layers = sum(layers)
        start_in_planes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = NetworkBlock(block, start_in_planes, 16*width, layers[0], 0)
        self.layer2 = NetworkBlock(block, 16*width, 32*width, layers[1], layers[0], stride=2)
        self.layer3 = NetworkBlock(block, 32*width, 64*width, layers[2], layers[0]+layers[1], stride=2)
        self.bn2 = nn.BatchNorm2d(64*width)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64*width, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn2(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        
    def forward_hl(self, x):
        layer_dict = {}
        x = self.conv1(x)
        x = self.bn1(x)

        layer_dict["conv1"] = x
        out = self.relu(x)

        out, layer_dict_1 = self.layer1.forward_hl(out)
        out, layer_dict_2 = self.layer2.forward_hl(out)
        out, layer_dict_3 = self.layer3.forward_hl(out)

        bn = self.bn2(out)
        out = self.relu(bn)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        layer_dict.update(layer_dict_1)
        layer_dict.update(layer_dict_2)
        layer_dict.update(layer_dict_3)

        return out, layer_dict

def bn_wideresnet16(**kwargs):
    model = BNWideResNet(BasicBlock, [2, 2, 2], **kwargs)
    return model
