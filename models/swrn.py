import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from layers import *

class PreprocessBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super(PreprocessBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.convShortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        out = self.relu(self.bn(x))
        return [self.convShortcut(out), self.conv(out)]

class BNReluConv(nn.Module):
    def __init__(self, planes, bank=None):
        super(BNReluConv, self).__init__()
        self.bank = bank
        self.bn = nn.BatchNorm2d(planes)
        if self.bank: self.conv = SConv2d(self.bank)
        else: self.conv = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, coefficient=None):
        if self.bank: return self.conv(self.relu(self.bn(x)), coefficient)
        else: return self.conv(self.relu(self.bn(x)))

# class Block(nn.Module):
#     def __init__(self, in_planes, out_planes, stride, bank=None):
#         super(Block, self).__init__()
#         self.bank = bank
#
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         if self.bank: self.conv1 = SConv2d(self.bank)
#         else: self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
#
#         self.bn2 = nn.BatchNorm2d(out_planes)
#         if self.bank: self.conv2 = SConv2d(self.bank)
#         else: self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.equalInOut = (in_planes == out_planes)
#         self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None
#
#     def forward(self, x):
#         residual = x
#         out = self.relu(self.bn1(x))
#         if not self.equalInOut: residual = out
#         out = self.conv2(self.relu(self.bn2(self.conv1(out))))
#         if self.convShortcut is not None: residual = self.convShortcut(residual)
#         return out + residual

class DynamicStage(nn.Module):
    def __init__(self, in_planes, out_planes, num_blocks, num_templates, stride=1):
        super(DynamicStage, self).__init__()
        self.nlayers = num_blocks * 2 - 1
        self.num_templates = num_templates
        self.bank = TemplateBank(num_templates, out_planes, out_planes, 3) if num_templates > 0 else None
        self.add_module('preprocess', PreprocessBlock(in_planes, out_planes, stride))
        for i in range(0, self.nlayers): self.add_module('conv'+str(i), BNReluConv(out_planes, self.bank))

        self.wrn_adj = torch.zeros(self.nlayers+3, self.nlayers+3).requires_grad_(False)
        for i in range(2, self.nlayers+3):
            if i % 2 == 0:
                self.wrn_adj[i, i-1] = 1.
            else:
                self.wrn_adj[i, np.arange(int(i/2)+1)*2] = 1.
        self.wrn_adj = self.wrn_adj.cuda()
        print(self.wrn_adj)

    # adjacency of size ((nlayers+3) * (nlayers+3))
    def forward(self, x, adjacency=None, invalid_layers=None, coefficients=None):
        if not type(adjacency) == torch.Tensor:
            adjacency = self.wrn_adj
            invalid_layers = []
        outputs = self.preprocess(x)
        for i in range(self.nlayers):
            outputs.append(None if i+2 in invalid_layers else getattr(self, 'conv'+str(i))(sum(outputs[j]*adjacency[i+2, j] for j in range(i+2) if j not in invalid_layers), coefficients[i] if self.num_templates else None))
        return sum(outputs[j]*adjacency[self.nlayers+2, j] for j in range(self.nlayers+2) if j not in invalid_layers)

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

        # self.bank_1 = TemplateBank(self.num_templates, n_channels[1], n_channels[1], 3)
        # self.stage_1 = self._make_layer(n_channels[0], n_channels[1], num_blocks, self.bank_1, 1)
        self.stage_1 = DynamicStage(n_channels[0], n_channels[1], num_blocks, num_templates, 1)

        # self.bank_2 = TemplateBank(self.num_templates, n_channels[2], n_channels[2], 3)
        # self.stage_2 = self._make_layer(n_channels[1], n_channels[2], num_blocks, self.bank_2, 2)
        self.stage_2 = DynamicStage(n_channels[1], n_channels[2], num_blocks, num_templates, 2)

        # self.bank_3 = TemplateBank(self.num_templates, n_channels[3], n_channels[3], 3)
        # self.stage_3 = self._make_layer(n_channels[2], n_channels[3], num_blocks, self.bank_3, 2)
        self.stage_3 = DynamicStage(n_channels[2], n_channels[3], num_blocks, num_templates, 2)

        self.lastact = nn.Sequential(nn.BatchNorm2d(n_channels[3]), nn.ReLU(inplace=True))
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

    # def _make_layer(self, in_planes, out_planes, num_blocks, bank, stride=1):
    #     blocks = []
    #     blocks.append(Block(in_planes, out_planes, stride))
    #     for i in range(1, num_blocks): blocks.append(Block(out_planes, out_planes, 1, bank))
    #     return nn.Sequential(*blocks)

    def forward(self, x, adjacencies=None, invalid_layerss=None, coefficients=None):
        x = self.conv_3x3(x)
        x = self.stage_1(x, adjacencies[0] if adjacencies else None, invalid_layerss[0] if invalid_layerss else None, coefficients[0] if coefficients else None)
        x = self.stage_2(x, adjacencies[1] if adjacencies else None, invalid_layerss[1] if invalid_layerss else None, coefficients[1] if coefficients else None)
        x = self.stage_3(x, adjacencies[2] if adjacencies else None, invalid_layerss[2] if invalid_layerss else None, coefficients[2] if coefficients else None)
        x = self.lastact(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def swrn(depth, width, num_templates, num_classes=10):
    model = SWRN(depth, width, num_templates, num_classes)
    return model
