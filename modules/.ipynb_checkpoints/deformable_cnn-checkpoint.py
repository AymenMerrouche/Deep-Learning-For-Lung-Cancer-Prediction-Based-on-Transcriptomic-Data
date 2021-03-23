from __future__ import absolute_import, division

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from modules.deform_conv import th_batch_map_offsets, th_generate_grid
import copy

class ConvOffset2D(nn.Conv2d):
    """ConvOffset2D
    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation
    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """
    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init
        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """
        self.filters = filters
        self._grid_param = None
        super(ConvOffset2D, self).__init__(self.filters, self.filters*2, 3, padding=1, bias=False, **kwargs)
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))

    def forward(self, x):
        """Return the deformed featured map"""
        x_shape = x.size()
        offsets = super(ConvOffset2D, self).forward(x)

        # offsets: (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)

        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)

        # X_offset: (b*c, h, w)
        x_offset = th_batch_map_offsets(x, offsets, grid=self._get_grid(self,x))

        # x_offset: (b, h, w, c)
        x_offset = self._to_b_c_h_w(x_offset, x_shape)

        return x_offset

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_height, input_width = x.size(0), x.size(1), x.size(2)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_height, input_width, dtype, cuda)
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x
    

class DeformConvNet(nn.Module):
    def __init__(self):
        super(DeformConvNet, self).__init__()
        
        # conv11
        self.conv11 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(8)

        # conv12
        self.offset12 = ConvOffset2D(8)
        self.conv12 = nn.Conv2d(8, 12, 3, padding=1, stride=2)
        self.bn12 = nn.BatchNorm2d(12)

        # conv21
        self.offset21 = ConvOffset2D(12)
        self.conv21 = nn.Conv2d(12, 12, 3, padding= 1)
        self.bn21 = nn.BatchNorm2d(12)

        # out
        self.fc = nn.Linear(12, 2)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.bn11(x)
        
        x = self.offset12(x)
        x = F.relu(self.conv12(x))
        x = self.bn12(x)
        
        x = self.offset21(x)
        x = F.relu(self.conv21(x))
        x = self.bn21(x)
        

        
        x = F.avg_pool2d(x, kernel_size=[x.size(2), x.size(3)])
        x = self.fc(x.view(x.size()[:2]))
        return x

    def freeze(self, module_classes):
        '''
        freeze modules for finetuning
        '''
        for k, m in self._modules.items():
            if any([type(m) == mc for mc in module_classes]):
                for param in m.parameters():
                    param.requires_grad = False

    def unfreeze(self, module_classes):
        '''
        unfreeze modules
        '''
        for k, m in self._modules.items():
            if any([isinstance(m, mc) for mc in module_classes]):
                for param in m.parameters():
                    param.requires_grad = True

    def parameters(self):
        return filter(lambda p: p.requires_grad, super(DeformConvNet, self).parameters())

def get_cnn():
    return ConvNet()

def get_deform_cnn_fine_tune(pretrained_model, freeze_cnn=True, freeze_filter=[nn.Conv2d]):
    model = copy.deepcopy(pretrained_model)
    if freeze_cnn:
        model.freeze(freeze_filter)
    model.fc = nn.Linear(12, 2)
    return model