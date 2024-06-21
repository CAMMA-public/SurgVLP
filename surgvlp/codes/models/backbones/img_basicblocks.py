import torch.nn as nn
import torch as th
from codes.models.utils import get_padding_shape

class InceptionBlock(nn.Module):

    def __init__(self, input_dim,
                num_outputs_0_0a,
                num_outputs_1_0a,
                num_outputs_1_0b,
                num_outputs_2_0a,
                num_outputs_2_0b,
                num_outputs_3_0b,
                gating=True):
        super(InceptionBlock, self).__init__()
        self.conv_b0 = STConv3D(input_dim, num_outputs_0_0a, [1, 1, 1])
        self.conv_b1_a = STConv3D(input_dim, num_outputs_1_0a, [1, 1, 1])
        self.conv_b1_b = STConv3D(num_outputs_1_0a, num_outputs_1_0b, [3, 3, 3],
                                    padding=1, separable=True)
        self.conv_b2_a = STConv3D(input_dim, num_outputs_2_0a, [1, 1, 1])
        self.conv_b2_b = STConv3D(num_outputs_2_0a, num_outputs_2_0b, [3, 3, 3],
                                    padding=1, separable=True)
        self.maxpool_b3 = th.nn.MaxPool3d((3, 3, 3), stride=1, padding=1)
        self.conv_b3_b = STConv3D(input_dim, num_outputs_3_0b, [1, 1, 1])
        self.gating = gating
        self.output_dim = num_outputs_0_0a + num_outputs_1_0b +\
                    num_outputs_2_0b + num_outputs_3_0b
        if gating:
            self.gating_b0 = SelfGating(num_outputs_0_0a)
            self.gating_b1 = SelfGating(num_outputs_1_0b)
            self.gating_b2 = SelfGating(num_outputs_2_0b)
            self.gating_b3 = SelfGating(num_outputs_3_0b)

    def forward(self, input):
      """Inception block
      """
      b0 = self.conv_b0(input)
      b1 = self.conv_b1_a(input)
      b1 = self.conv_b1_b(b1)
      b2 = self.conv_b2_a(input)
      b2 = self.conv_b2_b(b2)
      b3 = self.maxpool_b3(input)
      b3 = self.conv_b3_b(b3)
      if self.gating:
          b0 = self.gating_b0(b0)
          b1 = self.gating_b1(b1)
          b2 = self.gating_b2(b2)
          b3 = self.gating_b3(b3)
      return th.cat((b0, b1, b2, b3), dim=1)

class SelfGating(nn.Module):

    def __init__(self, input_dim):
        super(SelfGating, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, input_tensor):
      """Feature gating as used in S3D-G.
      """
      spatiotemporal_average = th.mean(input_tensor, dim=[2, 3, 4])
      weights = self.fc(spatiotemporal_average)
      weights = th.sigmoid(weights)
      return weights[:, :, None, None, None] * input_tensor


class STConv3D(nn.Module):

    def __init__(self,
                input_dim,
                output_dim,
                kernel_size,
                stride=1,
                padding=0,
                separable=False):
        super(STConv3D, self).__init__()
        self.separable = separable
        self.relu = nn.ReLU(inplace=True)
        assert len(kernel_size) == 3
        if separable and kernel_size[0] != 1:
            spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
            temporal_kernel_size = [kernel_size[0], 1, 1]
            if isinstance(stride, list) and len(stride) == 3:
              spatial_stride = [1, stride[1], stride[2]]
              temporal_stride = [stride[0], 1, 1]
            else:
              spatial_stride = [1, stride, stride]
              temporal_stride = [stride, 1, 1]
            if isinstance(padding, list) and len(padding) == 3:
              spatial_padding = [0, padding[1], padding[2]]
              temporal_padding = [padding[0], 0, 0]
            else:
              spatial_padding = [0, padding, padding]
              temporal_padding = [padding, 0, 0]
        if separable:
            self.conv1 = nn.Conv3d(input_dim, output_dim,
                                   kernel_size=spatial_kernel_size,
                                   stride=spatial_stride,
                                   padding=spatial_padding, bias=False)
            self.bn1 = nn.BatchNorm3d(output_dim)
            self.conv2 = nn.Conv3d(output_dim, output_dim,
                                   kernel_size=temporal_kernel_size,
                                   stride=temporal_stride,
                                   padding=temporal_padding, bias=False)
            self.bn2 = nn.BatchNorm3d(output_dim)
        else:
            self.conv1 = nn.Conv3d(input_dim, output_dim,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=padding, bias=False)
            self.bn1 = nn.BatchNorm3d(output_dim)


    def forward(self, input):
        out = self.relu(self.bn1(self.conv1(input)))
        if self.separable:
            out = self.relu(self.bn2(self.conv2(out)))
        return out


class MaxPool3dTFPadding(nn.Module):
    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = th.nn.ConstantPad3d(padding_shape, 0)
        self.pool = th.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out