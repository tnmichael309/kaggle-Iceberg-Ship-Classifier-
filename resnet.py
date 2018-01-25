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

class DropoutBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(DropoutBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop = nn.Dropout2d(p=0.05)
        
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
 

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduction),
            # nn.ReLU(inplace=True),
            nn.PReLU(),
            nn.Linear(reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
        
class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
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
        out = self.se(out)

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
        self.drop2 = nn.Dropout2d(p=0.2)
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.drop(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.drop2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        #out = self.drop2(out)
        
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

def paperResNet18():
	return PaperResNet18(DropoutBasicBlock, [2, 2, 2, 2])
    
class PaperResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        self.inplanes = 32
        super(PaperResNet18, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        # for inc_angle feature
        n_meta_features = 1 #1: 'inc_angle', 1001: 'transfer learning features included'
        self.bn2 = nn.BatchNorm1d(n_meta_features, momentum=0.1)
        
        self.fc = nn.Linear(256 + n_meta_features, 1)
        
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
        #print(imgs)
        
        inc_angles = self.bn2(inc_angles)
		
        # should concat along dim = 1, imgs = (batch_size, 2048), inc_angles = (batch size, 1)
        outputs = torch.cat([imgs, inc_angles], dim=1)
        #print(imgs)
        #print(imgs) #128x256x4x4
        outputs = self.fc(outputs)
                
        return self.prob(outputs)
        
class LessFilterResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1, n_meta_features=1, fc_output=False):
        self.inplanes = 64
        super(LessFilterResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3, ceil_mode=True)
        
        # for inc_angle feature
        #n_meta_features = 1 #1: 'inc_angle', 1001: 'transfer learning features included'
        if n_meta_features > 0:
            self.bn2 = nn.BatchNorm1d(n_meta_features, momentum=0.1)
        
        self.n_meta_features = n_meta_features
        
        # for transfer learning features
        self.fc_output = fc_output
        
        if self.fc_output is False:
            self.fc = nn.Linear(2048 + n_meta_features, 1)
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

    def forward(self, imgs, other_features=None):
        
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

        # should concat along dim = 1, imgs = (batch_size, 2048), inc_angles = (batch size, 1)
        if self.n_meta_features > 0:
            other_features = self.bn2(other_features)
            imgs = torch.cat([imgs, other_features], dim=1)
        
        
        if self.fc_output is False:
            outputs = self.fc(imgs)
            return self.prob(outputs)
        else:
            return imgs
   
class ConstrastResnet(nn.Module):
    def __init__(self):
        super(ConstrastResnet, self).__init__()
 
        self.resnet_model_1 = lessFilterResNet50(meta_features=0, fc_output=True)
        self.resnet_model_2 = lessFilterResNet50(meta_features=0, fc_output=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(4096, momentum=0.1)
        self.fc = nn.Linear(4096, 4096)
        self.drop = nn.Dropout(p=0.1)
        
        self.bn3 = nn.BatchNorm1d(4096, momentum=0.1)
        self.fc2 = nn.Linear(4096, 256)
        
        self.bn4 = nn.BatchNorm1d(256, momentum=0.1)
        self.fc3 = nn.Linear(256, 1)
        
        self.prob = nn.Sigmoid()
        
        
        
        state = torch.load('Trained_model/resnet_origin_sample08_soft_pseudo_label_n_valid_2.db')
        print('epoch=', state['epoch'], 'best_loss=', state['best_loss'])
        
        pretrained_dict = state['state_dict']
        model_dict = self.resnet_model_1.state_dict()
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        model_dict.update(pretrained_dict)
        self.resnet_model_1.load_state_dict(model_dict)
        
        
        state = torch.load('Trained_model/resnet_origin_sample08_soft_pseudo_label_n_valid_3.db')
        print('epoch=', state['epoch'], 'best_loss=', state['best_loss'])
        
        pretrained_dict = state['state_dict']
        model_dict = self.resnet_model_2.state_dict()
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        model_dict.update(pretrained_dict)
        self.resnet_model_2.load_state_dict(model_dict)
        
    def forward(self, img1, img2):
        
        outputs = torch.cat([self.resnet_model_1(img1), self.resnet_model_2(img2)], dim=1)
        
        outputs = self.bn2(outputs)
        outputs = self.fc(outputs)
        outputs = self.relu(outputs)
        outputs = self.drop(outputs)
        
        outputs = self.bn3(outputs)
        outputs = self.fc2(outputs)
        outputs = self.relu(outputs)
        outputs = self.drop(outputs)
        
        outputs = self.bn4(outputs)
        outputs = self.fc3(outputs)
        
        return self.prob(outputs)
        
   
class EnsembleResnet(nn.Module):
    def __init__(self, ensemble_num=2, num_classes=1, n_meta_features=1):
        super(EnsembleResnet, self).__init__()
        
        self.ensemble_num = ensemble_num
        
        self.resnet_model_1 = lessFilterResNet50(meta_features=n_meta_features)
        self.resnet_model_2 = lessFilterResNet50(meta_features=n_meta_features)
        
        '''
        self.models = [lessFilterResNet50(meta_features=n_meta_features) for _ in range(ensemble_num)]
        
        for i in range(ensemble_num):
            self.add_module('resnet_' + str(i), self.models[i])
        '''
        
    def forward(self, imgs, other_features):
    
        '''
        outputs = [self.models[i](imgs, other_features) for i in range(self.ensemble_num)]
        outputs = torch.cat(outputs, dim=1)
        #print(outputs)
        
        outputs = self.models[0](imgs, other_features)
        outputs = outputs.expand(outputs.size(0), 1)
        #print(outputs)
        
        for i in range(1, self.ensemble_num):
            out = self.models[i](imgs, other_features)
            out = out.expand(out.size(0), 1)
            #print(out)
            outputs = torch.cat([outputs, out], dim=1)
        '''
        
        outputs = torch.cat([self.resnet_model_1(imgs, other_features), self.resnet_model_2(imgs, other_features)], dim=1)
        outputs = torch.mean(outputs, dim=1)
        outputs = outputs.view(outputs.size(0), -1)
        return outputs
        
class RegLessFilterResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1, n_meta_features=1):
        self.inplanes = 64
        super(RegLessFilterResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3, ceil_mode=True)
        
        # for inc_angle feature
        #n_meta_features = 1 #1: 'inc_angle', 1001: 'transfer learning features included'
        self.bn2 = nn.BatchNorm1d(n_meta_features, momentum=0.1)
        
        # +1 for 'inc_angle' feature
        #self.fc = nn.Linear(2048 + n_meta_features, num_classes)
        
        # for transfer learning features
        self.fc = nn.Linear(2048 + n_meta_features, 1)
        
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
        #print(imgs)
        #print(imgs) #128x256x4x4
        outputs = self.fc(outputs)
        #outputs = self.bn3(outputs)
        #outputs = self.relu(outputs)
        #outputs = self.drop(outputs)
        #outputs = self.fc2(outputs)
        
        return outputs
        
class LessFilterResNetFeatureExtract(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        self.inplanes = 64
        super(LessFilterResNetFeatureExtract, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3, ceil_mode=True)
        
        # for inc_angle feature
        n_meta_features = 1 #1: 'inc_angle', 1001: 'transfer learning features included'
        self.bn2 = nn.BatchNorm1d(n_meta_features, momentum=0.1)
        
        # +1 for 'inc_angle' feature
        #self.fc = nn.Linear(2048 + n_meta_features, num_classes)
        
        # for transfer learning features
        self.fc = nn.Linear(2048 + n_meta_features, 1)
        self.bn3 = nn.BatchNorm1d(256, momentum=0.1)
        self.drop = nn.Dropout2d(p=0.1)
        self.fc2 = nn.Linear(256, num_classes)
        
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
        imgs = imgs.view(imgs.size(0), -1)
        
        inc_angles = self.bn2(inc_angles)
		
        # should concat along dim = 1, imgs = (batch_size, 2048), inc_angles = (batch size, 1)
        outputs = torch.cat([imgs, inc_angles], dim=1)
        
        return outputs
        
      
class NNResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1, n_meta_features=1):
        self.inplanes = 64
        super(NNResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3, ceil_mode=True)
        
        # for inc_angle feature
        #n_meta_features = 1 #1: 'inc_angle', 1001: 'transfer learning features included'
        
        # +1 for 'inc_angle' feature
        #self.fc = nn.Linear(2048 + n_meta_features, num_classes)
        
        # for transfer learning features
        self.bn2 = nn.BatchNorm1d(2048 + n_meta_features, momentum=0.1)
        self.fc = nn.Linear(2048 + n_meta_features, 256)
        self.drop = nn.Dropout2d(p=0.1)
        
        self.bn3 = nn.BatchNorm1d(256, momentum=0.1)
        self.fc2 = nn.Linear(256, 64)
        
        self.bn4 = nn.BatchNorm1d(64, momentum=0.1)
        self.fc3 = nn.Linear(64, num_classes)
        
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
        
        #inc_angles = self.bn2(inc_angles)
		
        # should concat along dim = 1, imgs = (batch_size, 2048), inc_angles = (batch size, 1)
        outputs = torch.cat([imgs, inc_angles], dim=1)
        #print(imgs)
        #print(imgs) #128x256x4x4
        outputs = self.bn2(outputs)
        outputs = self.fc(outputs)
        outputs = self.relu(outputs)
        outputs = self.drop(outputs)
        
        outputs = self.bn3(outputs)
        outputs = self.fc2(outputs)
        outputs = self.relu(outputs)
        outputs = self.drop(outputs)
        
        outputs = self.bn4(outputs)
        outputs = self.fc3(outputs)
        
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
def lessFilterResNet50(pretrained=False, skip_headers = None, meta_features=1, fc_output=False):
    model = LessFilterResNet(DropBottleneck, [3, 4, 6, 3], n_meta_features=meta_features, fc_output=fc_output)
    return model

def get_ensemble_resnet(ensemble_num=2, num_classes=1, meta_features=1):
    model = EnsembleResnet(ensemble_num=ensemble_num, n_meta_features=meta_features, num_classes=num_classes)
    return model
    
def nnResNet(meta_features=1):
	return NNResNet(DropBottleneck, [3, 4, 6, 3], n_meta_features=meta_features)

def regLessFilterResNet50(pretrained=False, skip_headers = None, meta_features=1):
    model = RegLessFilterResNet(DropBottleneck, [3, 4, 6, 3], n_meta_features=meta_features)
    return model
    
def lessFilterResNet50FeatureExtract():
    return LessFilterResNetFeatureExtract(DropBottleneck, [3, 4, 6, 3])
 
def SEResNet50(pretrained=False, skip_headers = None, meta_features=1):
    model = LessFilterResNet(SEBottleneck, [3, 4, 6, 3], n_meta_features=meta_features)
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