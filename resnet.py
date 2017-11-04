import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
        
# modification:
#  add momentum = .9 to bn
#  add dropout after relu 
class DropBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(DropBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.drop = nn.Dropout2d(p=0.2)
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.drop(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.drop(out)
        
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        out = self.drop(out)
        
        return out

class WideDropBottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(WideDropBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * WideDropBottleneck.expansion, kernel_size=3, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes * WideDropBottleneck.expansion, momentum=0.9)
        
        self.conv2 = nn.Conv2d(planes * WideDropBottleneck.expansion, planes * WideDropBottleneck.expansion, kernel_size=3, stride=1, # change
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * WideDropBottleneck.expansion, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.drop = nn.Dropout2d(p=0.2)
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
            
        out = self.drop(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.drop(out)
        
        return out
        
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x    
  
class FineTuneResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(FineTuneResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # To train features
        n_meta_features = 1
        self.bn2 = nn.BatchNorm1d(n_meta_features, momentum=0.1)
        self.fc2 = nn.Linear(num_classes+n_meta_features, 30)
        self.fc3 = nn.Linear(30, 1)
        self.prob = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.in_features * m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, imgs, other_features):
        inc_angles = other_features
        
        imgs = self.conv1(imgs)
        imgs = self.bn1(imgs)
        imgs = self.relu(imgs)
        imgs = self.maxpool(imgs)

        imgs = self.layer1(imgs)
        imgs = self.layer2(imgs)
        imgs = self.layer3(imgs)
        imgs = self.layer4(imgs)
        imgs = self.avgpool(imgs)
        imgs = imgs.view(imgs.size(0), -1)
        imgs = self.fc(imgs)
        
        inc_angles = self.bn2(inc_angles)
        # should concat along dim = 1, imgs = (batch_size, 2048), inc_angles = (batch size, 1)
        outputs = torch.cat([imgs, inc_angles], dim=1)
        #print(imgs) #128x256x4x4
        outputs = self.fc2(outputs)
        outputs = self.fc3(outputs)
         
        return self.prob(outputs)
        
class LessFilterResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        self.inplanes = 32
        super(LessFilterResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3, ceil_mode=True)
        
        # for inc_angle feature
        n_meta_features = 1
        self.bn2 = nn.BatchNorm1d(n_meta_features, momentum=0.1)
        
        # +1 for 'inc_angle' feature
        self.fc = nn.Linear(256 + n_meta_features, num_classes)
        
        self.prob = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.in_features * m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if block is not WideDropBottleneck:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=3, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, imgs, other_features):
        inc_angles = other_features
        
        imgs = self.conv1(imgs)
        imgs = self.bn1(imgs)
        imgs = self.relu(imgs)
        imgs = self.maxpool(imgs)

        imgs = self.layer1(imgs)
        imgs = self.layer2(imgs)
        imgs = self.layer3(imgs)
        imgs = self.layer4(imgs)
        imgs = self.avgpool(imgs)
        imgs = imgs.view(imgs.size(0), -1)
        
        inc_angles = self.bn2(inc_angles)
        # should concat along dim = 1, imgs = (batch_size, 2048), inc_angles = (batch size, 1)
        outputs = torch.cat([imgs, inc_angles], dim=1)
        #print(imgs) #128x256x4x4
        outputs = self.fc(outputs)
         
        return self.prob(outputs)

def wideResNet():
    model = ResNet(WideDropBottleneck, [2, 2, 2, 2])
    return model

# return what's in dict1 but not in dict2
def diff_dict(dict1, dict2):
    ret_dict = {}
    for k,v in dict1.items():
        if k not in dict2:
            ret_dict[k] = v
            
    return ret_dict

# if pretrained is True,
# not only return the model, but also the unfrozen dict to add to param group in the optimizer
# 'unfrozen dict': parameters to be trained    
def fineTuneResNet50(pretrained=False, skip_headers = None):
    model = FineTuneResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        
        used_dict = {}
        for k, v in pretrained_dict.items():
            if k not in model_dict:
                continue
            
            should_skip = False
                
            if skip_headers is not None:
                for header in skip_headers:
                    if header == k[:len(header)]:
                        should_skip = True
                        break
                
            if should_skip is False:
                used_dict[k] = v
        
        model_dict.update(used_dict)
        
        print("\n\n==== Pre-trained layers ==== \n\n")
        for k in pretrained_dict:
            print(k)
            
        print("\n\n====  Loaded layers ==== \n\n")
        for k in used_dict:
            print(k)
            
        model.load_state_dict(model_dict)
        unused_dict = diff_dict(model_dict, used_dict)
        
        # return the model, and the unfrozen dict to add to param group in the optimizer
        # 'unfrozen dict': parameters to be trained
        return model, unused_dict
        
    else:
        return model
        
# if pretrained is True,
# not only return the model, but also the unfrozen dict to add to param group in the optimizer
# 'unfrozen dict': parameters to be trained    
def lessFilterResNet50(pretrained=False, skip_headers = None):
    model = LessFilterResNet(DropBottleneck, [3, 4, 6, 3])
    return model
    
def resnet18(pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [1, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model