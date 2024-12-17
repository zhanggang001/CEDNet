import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import NECKS
from mmcv.cnn import build_norm_layer


class Conv2dStaticSamePadding(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x


class MaxPool2dStaticSamePadding(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x


class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):

    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SeparableConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels=None, norm_cfg=None, activation=False):
        super(SeparableConvBlock, self).__init__()

        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm_cfg is not None
        if self.norm:
            self.bn = build_norm_layer(norm_cfg, out_channels)[1]

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class BiFPNBlock(nn.Module):

    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, norm_cfg=None):
        super(BiFPNBlock, self).__init__()

        self.epsilon = epsilon

        # Conv layers
        self.conv6_up = SeparableConvBlock(num_channels, norm_cfg=norm_cfg)
        self.conv5_up = SeparableConvBlock(num_channels, norm_cfg=norm_cfg)
        self.conv4_up = SeparableConvBlock(num_channels, norm_cfg=norm_cfg)
        self.conv3_up = SeparableConvBlock(num_channels, norm_cfg=norm_cfg)
        self.conv4_down = SeparableConvBlock(num_channels, norm_cfg=norm_cfg)
        self.conv5_down = SeparableConvBlock(num_channels, norm_cfg=norm_cfg)
        self.conv6_down = SeparableConvBlock(num_channels, norm_cfg=norm_cfg)
        self.conv7_down = SeparableConvBlock(num_channels, norm_cfg=norm_cfg)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.swish = MemoryEfficientSwish()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                build_norm_layer(norm_cfg, num_channels)[1],
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                build_norm_layer(norm_cfg, num_channels)[1],
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                build_norm_layer(norm_cfg, num_channels)[1],
            )

            self.p5_to_p6 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                build_norm_layer(norm_cfg, num_channels)[1],
                MaxPool2dStaticSamePadding(3, 2)
            )
            self.p6_to_p7 = nn.Sequential(
                MaxPool2dStaticSamePadding(3, 2)
            )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                build_norm_layer(norm_cfg, num_channels)[1],
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                build_norm_layer(norm_cfg, num_channels)[1],
            )

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)))

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up)))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out


@NECKS.register_module()
class BiFPN(nn.Module):

    def __init__(self, channels=256, input_channels=[192, 384, 768], num_layers=2, epsilon=1e-4,
                 norm_cfg=dict(type='BN', momentum=0.01, eps=1e-3)):
        super(BiFPN, self).__init__()

        bifpns = [BiFPNBlock(channels, input_channels, first_time=True, epsilon=epsilon, norm_cfg=norm_cfg)]
        for _ in range(num_layers - 1):
            bifpns.append(BiFPNBlock(channels, input_channels, first_time=False, epsilon=epsilon, norm_cfg=norm_cfg))
        self.bifpn = nn.Sequential(*bifpns)

    def init_weights(self):
        pass

    def forward(self, inputs):
        return self.bifpn(inputs[1:])


class BiFPNLayer(nn.Module):

    def __init__(self, dims=[256, 256, 256], blocks=[1, 1, 1], norm_cfg=None):
        super().__init__()

        self.decoder_upsample = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dims[2], dims[1], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)),
            nn.Sequential(nn.Conv2d(dims[1], dims[0], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        ])

        self.decoder = nn.Sequential(
            nn.Sequential(*[SeparableConvBlock(dims[1], norm_cfg=norm_cfg) for _ in range(blocks[1])]),
            nn.Sequential(*[SeparableConvBlock(dims[0], norm_cfg=norm_cfg) for _ in range(blocks[0])]),
        )

        self.encoder_downsample = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2), build_norm_layer(norm_cfg, dims[1])[1]),
            nn.Sequential(nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2), build_norm_layer(norm_cfg, dims[2])[1]),
        ])

        self.encoder = nn.ModuleList([
            nn.Sequential(*[SeparableConvBlock(dims[1], norm_cfg=norm_cfg) for _ in range(blocks[1])]),
            nn.Sequential(*[SeparableConvBlock(dims[2], norm_cfg=norm_cfg) for _ in range(blocks[2])]),
        ])

    def forward(self, input):
        c3, c4, c5 = input
        c4 = self.decoder[0](self.decoder_upsample[0](c5) + c4)
        c3 = self.decoder[1](self.decoder_upsample[1](c4) + c3)

        p3 = c3
        p4 = self.encoder[1](self.encoder_downsample[0](p3) + c4)
        p5 = self.encoder[2](self.encoder_downsample[1](p4) + c5)
        return p3, p4, p5


@NECKS.register_module()
class BiFPNScratch(nn.Module):

    def __init__(self,
                 input_channels=[192, 384, 768],
                 dims=[256, 256, 256],
                 blocks=[1, 1, 1],
                 num_layers=2,
                 norm_cfg=dict(type='GN', )):
        super(BiFPN, self).__init__()

        self.lateral_convs = nn.ModuleList([nn.Conv1d(input_channels[idx], dims[idx], 1) for idx in range(len(dims))])
        self.bifpn = nn.Sequential(*[BiFPNLayer(dims, blocks, norm_cfg) for _ in range(num_layers)])

    def init_weights(self):
        pass

    def forward(self, inputs):
        reduce_input = [conv(x) for conv, x in zip(self.lateral_convs, inputs[1:])]
        return self.bifpn(reduce_input)
