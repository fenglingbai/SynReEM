
#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import sys
# 将当前路径添加到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional
import torch.nn.functional as F

class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super(ConvDropoutNormNonlin, self).__init__()

        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, bias=True)
        self.instnorm = nn.InstanceNorm3d(output_channels, eps=1e-5, affine=True)
        self.lrelu = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return self.lrelu(self.instnorm(x))

class resBlock_pni(nn.Module):
    # https://github.com/torms3/Superhuman/blob/torch-0.4.0/code/rsunet.py#L145
    def __init__(self, in_planes, out_planes):
        super(resBlock_pni, self).__init__()
        self.block1 = ConvDropoutNormNonlin(in_planes, out_planes, 
                                           kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.block2 = ConvDropoutNormNonlin(out_planes, out_planes, 
                                           kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.block3 = ConvDropoutNormNonlin(out_planes, out_planes, 
                                           kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.block4 = nn.InstanceNorm3d(out_planes, eps=1e-5, affine=True)
        self.block5 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
    def forward(self, x):
        residual = self.block1(x)
        out = residual + self.block3(self.block2(residual))
        out = self.block5(self.block4(out))
        return out

class SegHead(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, aemc_channels, output_channels):
        super(SegHead, self).__init__()

        assert input_channels % 2 == 0

        self.conv_aemc1 = nn.Conv3d(input_channels, input_channels // 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), 
                                    padding=(1, 2, 2), dilation=(1, 2, 2), bias=True)
        self.instnorm_aemc1 = nn.InstanceNorm3d(input_channels // 2, eps=1e-5, affine=True)
        self.lrelu_aemc1 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

        self.conv_aemc2 = nn.Conv3d(input_channels, aemc_channels, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False)

        self.conv_sem1 = nn.Conv3d(input_channels, input_channels // 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), 
                                    padding=(1, 1, 1), dilation=(1, 1, 1), bias=True)
        self.instnorm_sem1 = nn.InstanceNorm3d(input_channels // 2, eps=1e-5, affine=True)
        self.lrelu_sem1 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        self.conv_sem2 = nn.Conv3d(input_channels // 2, output_channels, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False)



    def forward(self, x):
        aemc_x1 = self.lrelu_aemc1(self.instnorm_aemc1(self.conv_aemc1(x)))
        sem_x1 = self.lrelu_sem1(self.instnorm_sem1(self.conv_sem1(x)))

        aemc_x2 = self.conv_aemc2(torch.cat((aemc_x1, sem_x1), dim=1))
        sem_x2 = self.conv_sem2(sem_x1)

        return aemc_x2, sem_x2

class SynReEM(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels=1, 
                num_features=[28, 36, 48, 64, 80], 
                # num_features=[32, 48, 96, 128, 160], 
                num_classes=5, 
                seman_channels=2, 
                deep_supervision=True,
                test_mode=False):

        super(SynReEMDualV2Aux, self).__init__()
        # ----------------------------------------parameters of NNUNet
        self.conv_op = nn.Conv3d
        self._deep_supervision = self.do_ds = deep_supervision
        self.num_classes = num_classes
        self.seman_channels = seman_channels
        self.test_mode = test_mode
        # ----------------------------------------

        #############################
        self.conv0 = ConvDropoutNormNonlin(input_channels, num_features[0], 
                                           kernel_size=(1, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2))
        self.conv1 = resBlock_pni(in_planes=num_features[0], out_planes=num_features[0])
        self.down1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = resBlock_pni(in_planes=num_features[0], out_planes=num_features[1])
        self.down2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3 = resBlock_pni(in_planes=num_features[1], out_planes=num_features[2])
        self.down3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv4 = resBlock_pni(in_planes=num_features[2], out_planes=num_features[3])
        self.down4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))


        self.center = resBlock_pni(in_planes=num_features[3], out_planes=num_features[4])
        # out_size = (in_size - 1) * S + K - 2P + output_padding
        self.up1 = nn.ConvTranspose3d(num_features[4], num_features[3], 
                                    kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0), output_padding=(0, 0, 0))
        self.conv5 = resBlock_pni(in_planes=num_features[3]*2, out_planes=num_features[3])

        self.up2 = nn.ConvTranspose3d(num_features[3], num_features[2], 
                                    kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0), output_padding=(0, 0, 0))
        self.conv6 = resBlock_pni(in_planes=num_features[2]*2, out_planes=num_features[2])

        self.up3 = nn.ConvTranspose3d(num_features[2], num_features[1], 
                                    kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0), output_padding=(0, 0, 0))
        self.conv7 = resBlock_pni(in_planes=num_features[1]*2, out_planes=num_features[1])

        self.up4 = nn.ConvTranspose3d(num_features[1], num_features[0], 
                                    kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0), output_padding=(0, 0, 0))
        self.conv8 = resBlock_pni(in_planes=num_features[0]*2, out_planes=num_features[0])

        self.conv9 = ConvDropoutNormNonlin(num_features[0], num_features[0], 
                                           kernel_size=(1, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2))

        self.outputs = nn.ModuleList(
            [SegHead(c, self.num_classes, output_channels=self.seman_channels) for c in [num_features[0], num_features[0], num_features[1]]]
        )

        #############################

        self.apply(InitWeights_He(1e-2))
            # self.apply(print_module_training_status)

    def forward(self, x):
        x1 = self.conv1(self.conv0(x))
        x2 = self.conv2(self.down1(x1))
        x3 = self.conv3(self.down2(x2))
        x4 = self.conv4(self.down3(x3))
        x5 = self.center(self.down4(x4))

        x6 = self.conv5(torch.cat((self.up1(x5), x4), dim=1))
        x7 = self.conv6(torch.cat((self.up2(x6), x3), dim=1))
        x8 = self.conv7(torch.cat((self.up3(x7), x2), dim=1))
        x9 = self.conv8(torch.cat((self.up4(x8), x1), dim=1))
        x10 = self.conv9(x9)

        ######################## !!!!!!
        if self._deep_supervision and self.do_ds:
            features = [x10, x9, x8]
            out_list = [self.outputs[i](features[i]) for i in range(3)]
            return tuple([out_list[0][0], out_list[1][0], out_list[2][0], out_list[0][1], out_list[1][1], out_list[2][1]])
            # return  tuple([self.conv_class[0](x9)])
        elif self.test_mode:
            return torch.cat((self.outputs[0](x10)[0], self.outputs[0](x10)[1]), dim=1)
        else:
            return self.outputs[0](x10)[0]


if __name__ == '__main__':
    SynReEM = SynReEM(input_channels=1, num_classes=5, seman_channels=2, deep_supervision=False, test_mode=True)
    # input = torch.randn((1, 1, 19, 255, 256))
    input = torch.randn((1, 1, 16, 320, 320))
    output = SynReEM(input)

    print([e.shape for e in output])
    from thop import profile
    flops, params = profile(SynReEM, inputs=(input, ))
    print('FLOPs (G): %.2f, Params (M): %.2f'%(flops/1e9, params/1e6)) #flops单位G，para单位M
    # 677.9469824 1.58878
    print('ok')

