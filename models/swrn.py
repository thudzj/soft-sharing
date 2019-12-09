import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from layers import *

class PreprocessBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super(PreprocessBlock, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        # self.convShortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        # self.bnShortcut = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        # out = self.relu(x)
        return self.bn(self.conv(self.relu(x)))#[self.bnShortcut(self.convShortcut(out)), self.bn(self.conv(out))]

class ReluConvBN(nn.Module):
    def __init__(self, planes, bank=None):
        super(ReluConvBN, self).__init__()
        self.bank = bank
        self.bn = nn.BatchNorm2d(planes)
        if self.bank: self.conv = SConv2d(self.bank)
        else: self.conv = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, coefficient=None):
        if self.bank: return self.bn(self.conv(self.relu(x), coefficient))
        else: return self.bn(self.conv(self.relu(x)))

class DynamicStage(nn.Module):
    def __init__(self, in_planes, out_planes, num_blocks, num_templates, stride=1):
        super(DynamicStage, self).__init__()
        self.nlayers = num_blocks * 2 - 1
        self.num_templates = num_templates
        self.bank = TemplateBank(num_templates, out_planes, out_planes, 3) if num_templates > 0 else None
        self.add_module('preprocess', PreprocessBlock(in_planes, out_planes, stride))
        for i in range(0, self.nlayers): self.add_module('conv'+str(i), ReluConvBN(out_planes, self.bank))

    def forward(self, x, adjacency=None, coefficients=None):
        in_layers, out_layers = [0], [self.nlayers-1]
        for i in range(0, self.nlayers):
            if i > 0 and adjacency[i].sum() == 0:
                in_layers.append(i)
            if i < self.nlayers-1 and adjacency[:, i].sum() == 0:
                out_layers.append(i)

        input_processed = self.preprocess(x)
        outputs = []
        for i in range(self.nlayers):
            outputs.append(getattr(self, 'conv'+str(i))(input_processed if i in in_layers else sum(outputs[j] for j in range(i) if adjacency[i, j] == 1), coefficients[i] if self.num_templates else None))
        return sum(outputs[j] for j in out_layers)

class SWRN(nn.Module):
    def __init__(self, depth, width, num_templates, num_classes):
        super(SWRN, self).__init__()

        n_channels = [16, 16*width, 32*width, 64*width]
        assert((depth - 4) % 6 == 0)
        num_blocks = (depth - 4) / 6
        print ('SWRN : Depth : {} , Widen Factor : {}, Templates per Group : {}'.format(depth, width, num_templates))

        self.num_classes = num_classes
        self.num_templates = num_templates

        self.conv_3x3 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(n_channels[0])

        self.stage_1 = DynamicStage(n_channels[0], n_channels[1], num_blocks, num_templates, 1)
        self.stage_2 = DynamicStage(n_channels[1], n_channels[2], num_blocks, num_templates, 2)
        self.stage_3 = DynamicStage(n_channels[2], n_channels[3], num_blocks, num_templates, 2)

        self.lastact = nn.ReLU(inplace=True)#nn.Sequential(nn.BatchNorm2d(n_channels[3]), nn.ReLU(inplace=True))
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(n_channels[3], num_classes)

        # for i in range(1,4):
        #     # initialize with identity
        #     coefficient_inits = torch.eye(layers_per_bank)
        #     sconv_group = filter(lambda (name, module): isinstance(module, SConv2d) and "stage_%s"%i in name, self.named_modules())
        #     for j, (name, module) in enumerate(sconv_group):
        #         print(name)
        #         module.coefficients.data = coefficient_inits[j].view_as(module.coefficients)

            # initialize with orthogonal
            # coefficient_inits = torch.zeros((layers_per_bank,num_templates,1,1,1,1))
            # nn.init.orthogonal_(coefficient_inits)
            # sconv_group = filter(lambda (name, module): isinstance(module, SConv2d) and "stage_%s"%i in name, self.named_modules())
            # for j, (name, module) in enumerate(sconv_group): module.coefficients.data = coefficient_inits[j]

            # initialize with sparse normal
            # coefficient_inits = torch.zeros((layers_per_bank,num_templates)).normal_()
            # zero_indices = np.concatenate([np.ones((layers_per_bank, int(num_templates/2))), np.zeros((layers_per_bank, int(num_templates/2)))], 1)
            # for tmp in range(layers_per_bank):
            #     np.random.shuffle(zero_indices[tmp])
            # coefficient_inits *= torch.from_numpy(zero_indices).float()
            # coefficient_inits /= coefficient_inits.sum(1, keepdim=True)
            # sconv_group = filter(lambda (name, module): isinstance(module, SConv2d) and "stage_%s"%i in name, self.named_modules())
            # for j, (name, module) in enumerate(sconv_group):
            #     module.coefficients.data = coefficient_inits[j].view_as(module.coefficients)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x, adjacencies, coefficients=None):
        adjacencies = adjacencies.squeeze().chunk(3)
        x = self.bn_1(self.conv_3x3(x))
        x = self.stage_1(x, adjacencies[0], coefficients[0] if coefficients else None)
        x = self.stage_2(x, adjacencies[1], coefficients[1] if coefficients else None)
        x = self.stage_3(x, adjacencies[2], coefficients[2] if coefficients else None)
        x = self.lastact(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def swrn(depth, width, num_templates, num_classes=10):
    model = SWRN(depth, width, num_templates, num_classes)
    return model
