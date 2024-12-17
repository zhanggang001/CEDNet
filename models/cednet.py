import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.registry import register_model

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import NORM_LAYERS


BaseBlock = None
FocalBlock = None
act_layer = None
ls_init_value = 1e-6


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first", "channels_first_v2"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

        elif self.data_format == "channels_first_v2":
            return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class Bottleneck(nn.Module):

    def __init__(self, dim, drop_path=0., norm_cfg=None, **kwargs):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            build_norm_layer(norm_cfg, dim)[1],
            act_layer(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1, stride=1, bias=False),
            build_norm_layer(norm_cfg, dim * 4)[1],
            act_layer(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, dim)[1],
        )

        self.act = act_layer()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.act(self.drop_path(x) + input)
        return x


class DBottleneck(nn.Module):

    def __init__(self, dim, drop_path=0., dilation=3, norm_cfg=None, **kwargs):
        super(DBottleneck, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            build_norm_layer(norm_cfg, dim)[1],
            act_layer(),
        )

        self.dwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3 * dilation, dilation=dilation, groups=dim),
            build_norm_layer(norm_cfg, dim)[1],
            act_layer())

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1, stride=1, bias=False),
            build_norm_layer(norm_cfg, dim * 4)[1],
            act_layer(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, dim)[1],
        )

        self.act = act_layer()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x) + x
        x = self.dwconv1(x) + x
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.act(self.drop_path(x) + input)
        return x


class NeXtBlock(nn.Module):

    def __init__(self, dim, drop_path=0., norm_cfg=None, **kwargs):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = build_norm_layer(norm_cfg, dim)[1]
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = act_layer()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            ls_init_value * torch.ones((dim)), requires_grad=True) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)    # input (N, C, *)

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class DNeXtBlock(nn.Module):

    def __init__(self, dim, drop_path=0., dilation=3, norm_cfg=None, **kwargs):
        super().__init__()

        self.dwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, dilation=1, groups=dim),
            build_norm_layer(norm_cfg, dim)[1],
            act_layer())

        self.dwconv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3 * dilation, dilation=dilation, groups=dim),
            build_norm_layer(norm_cfg, dim)[1],
            act_layer())

        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = act_layer()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            ls_init_value * torch.ones((dim)), requires_grad=True) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv1(x) + x
        x = self.dwconv2(x) + x

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinBlock(nn.Module):

    def __init__(self, dim, drop_path=0.0, window_size=7, norm_cfg=None, widx=0, **kwargs):
        super().__init__()

        self.norm_cfg = copy.deepcopy(norm_cfg)
        self.norm_cfg['data_format'] = "channels_last"
        num_heads = dim // 32

        self.dim = dim
        self.window_size = window_size
        self.shift_size = [0, window_size // 2][widx % 2]

        self.norm1 = build_norm_layer(self.norm_cfg, dim)[1]
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads)

        self.norm2 = build_norm_layer(self.norm_cfg, dim)[1]
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = act_layer()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn_mask = None

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)   # [N, C, H, W] -> [N, H, W, C]

        shortcut = x
        x = self.norm1(x)
        x = self.window_attention(x)
        x = shortcut + self.drop_path(x)

        shortcut = x
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = shortcut + self.drop_path(x)

        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        return x

    def window_attention(self, x):
        _, H, W, C = x.shape

        # # pad feature maps to multiples of window size
        # pad_l = pad_t = 0
        # pad_r = (self.window_size - W % self.window_size) % self.window_size
        # pad_b = (self.window_size - H % self.window_size) % self.window_size
        # x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))    # [B, Hp, Wp, C]
        # _, Hp, Wp, _ = x.shape

        # cyclic shift
        attn_mask = None
        if self.shift_size > 0 and H > self.window_size:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if self.attn_mask is None:
                self.attn_mask = self.calculate_attn_mask(H, W).to(x.device)
            attn_mask = self.attn_mask

        # partition windows
        x_windows = window_partition(x, self.window_size)  # [nW*B, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # [B, H, W, C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # if pad_r > 0 or pad_b > 0:
        #     x = x[:, :H, :W, :].contiguous()

        return x

    def calculate_attn_mask(self, H, W):
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, window_size, window_size, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


class DSwinBlock(SwinBlock):

    def __init__(self, dim, drop_path=0.0, dilation=3, window_size=7, norm_cfg=None, widx=0):
        super().__init__(dim, drop_path, window_size, norm_cfg, widx)

        self.norm_act1 = nn.Sequential(
            build_norm_layer(self.norm_cfg, dim)[1],
            act_layer())

        self.dwconv2 = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3 * dilation, dilation=dilation, groups=dim)

        self.norm_act2 = nn.Sequential(
            build_norm_layer(self.norm_cfg, dim)[1],
            act_layer())

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)   # [N, C, H, W] -> [N, H, W, C]

        shortcut = x
        x = self.norm1(x)
        x = self.norm_act1(self.window_attention(x)) + x
        x = self.norm_act2(self.dwconv2(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)) + x
        x = shortcut + self.drop_path(x)

        shortcut = x
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = shortcut + self.drop_path(x)

        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        return x


class Encoder(nn.Module):

    def __init__(self, dims=[192, 352, 512], blocks=[1, 1, 1], dp_rates=0., norm_cfg=None):
        super().__init__()

        assert isinstance(dp_rates, list)
        cum_sum = np.array([0] + blocks[:-1]).cumsum()

        self.encoder = nn.ModuleList([
            nn.Sequential(*[BaseBlock(dims[0], dp_rates[_], norm_cfg=norm_cfg, widx=_) for _ in range(blocks[0])]),
            nn.Sequential(*[BaseBlock(dims[1], dp_rates[cum_sum[1]+_], norm_cfg=norm_cfg, widx=_) for _ in range(blocks[1])]),
            nn.Sequential(*[FocalBlock(dims[2], dp_rates[cum_sum[2]+_], dilation=3, norm_cfg=norm_cfg, widx=_) for _ in range(blocks[2])]),
        ])

        self.encoder_downsample = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2), build_norm_layer(norm_cfg, dims[1])[1]),
            nn.Sequential(nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2), build_norm_layer(norm_cfg, dims[2])[1]),
        ])

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        c3 = self.encoder[0](x)
        c4 = self.encoder[1](self.encoder_downsample[0](c3))
        c5 = self.encoder[2](self.encoder_downsample[1](c4))
        return c3, c4, c5


class EncoderFPN(Encoder):

    def forward(self, x):
        if isinstance(x, tuple):
            c3 = self.encoder[0](x[0])
            c4 = self.encoder[1](x[1] + self.encoder_downsample[0](c3))
            c5 = self.encoder[2](x[2] + self.encoder_downsample[1](c4))
        else:
            c3 = self.encoder[0](x)
            c4 = self.encoder[1](self.encoder_downsample[0](c3))
            c5 = self.encoder[2](self.encoder_downsample[1](c4))
        return c3, c4, c5


class Decoder(nn.Module):

    def __init__(self, dims=[192, 352, 512], norm_cfg=None, **kwargs):
        super().__init__()

        self.decoder_upsample = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dims[2], dims[1], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)),
            nn.Sequential(nn.Conv2d(dims[1], dims[0], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        ])

        self.decoder_norm = nn.Sequential(
            build_norm_layer(norm_cfg, dims[1])[1],
            build_norm_layer(norm_cfg, dims[0])[1],
        )

    def forward(self, x):
        c3, c4, c5 = x
        c4 = self.decoder_norm[0](c4 + self.decoder_upsample[0](c5))
        c3 = self.decoder_norm[1](c3 + self.decoder_upsample[1](c4))
        return c3, c4, c5


class DecoderFPN(nn.Module):

    def __init__(self, dims=[192, 352, 512], norm_cfg=None, **kwargs):
        super().__init__()

        self.decoder_upsample = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dims[2], dims[1], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)),
            nn.Sequential(nn.Conv2d(dims[1], dims[0], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        ])

        self.decoder_norm = nn.Sequential(
            build_norm_layer(norm_cfg, dims[1])[1],
            build_norm_layer(norm_cfg, dims[0])[1],
        )

        self.decoder_conv = nn.Sequential(
            nn.Sequential(nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=1, padding=1), build_norm_layer(norm_cfg, dims[0])[1]),
            nn.Sequential(nn.Conv2d(dims[1], dims[1], kernel_size=3, stride=1, padding=1), build_norm_layer(norm_cfg, dims[1])[1]),
            nn.Sequential(nn.Conv2d(dims[2], dims[2], kernel_size=3, stride=1, padding=1), build_norm_layer(norm_cfg, dims[2])[1]),
        )

    def forward(self, x):
        c3, c4, c5 = x
        c4 = self.decoder_norm[0](c4 + self.decoder_upsample[0](c5))
        c3 = self.decoder_norm[1](c3 + self.decoder_upsample[1](c4))

        c3 = self.decoder_conv[0](c3)
        c4 = self.decoder_conv[1](c4)
        c5 = self.decoder_conv[2](c5)
        return c3, c4, c5


class DecoderUNet(nn.Module):

    def __init__(self, dims=[192, 352, 512], blocks=[1, 1], dp_rates=0., norm_cfg=None):
        super().__init__()

        assert isinstance(dp_rates, list)
        cum_sum = np.array([0] + blocks[:-1]).cumsum()

        self.decoder_upsample = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dims[2], dims[1], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)),
            nn.Sequential(nn.Conv2d(dims[1], dims[0], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        ])

        self.decoder_norm = nn.Sequential(
            build_norm_layer(norm_cfg, dims[1])[1],
            build_norm_layer(norm_cfg, dims[0])[1],
        )

        self.decoder = nn.Sequential(
            nn.Sequential(*[BaseBlock(dims[1], dp_rates[_], norm_cfg=norm_cfg, widx=_) for _ in range(blocks[0])]),
            nn.Sequential(*[BaseBlock(dims[0], dp_rates[cum_sum[0]+_], norm_cfg=norm_cfg, widx=_) for _ in range(blocks[1])]),
        )

    def forward(self, x):
        c3, c4, c5 = x
        c4 = self.decoder[0](self.decoder_norm[0](c4 + self.decoder_upsample[0](c5)))
        c3 = self.decoder[1](self.decoder_norm[1](c3 + self.decoder_upsample[1](c4)))
        return c3, c4, c5


class DecoderHourglass(nn.Module):

    def __init__(self, dims=[192, 352, 512], blocks=[1, 1], dp_rates=0., norm_cfg=None):
        super().__init__()

        assert isinstance(dp_rates, list)
        cum_sum = np.array([0] + blocks[:-1]).cumsum()

        self.decoder_upsample = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dims[2], dims[1], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)),
            nn.Sequential(nn.Conv2d(dims[1], dims[0], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        ])

        self.decoder_norm = nn.Sequential(
            build_norm_layer(norm_cfg, dims[1])[1],
            build_norm_layer(norm_cfg, dims[0])[1],
        )

        self.lateral = nn.Sequential(
            BaseBlock(dims[1], norm_cfg=norm_cfg, widx=0),
            BaseBlock(dims[0], norm_cfg=norm_cfg, widx=0),
        )

        self.decoder = nn.Sequential(
            nn.Sequential(*[BaseBlock(dims[1], dp_rates[_], norm_cfg=norm_cfg, widx=_) for _ in range(blocks[0])]),
            nn.Sequential(*[BaseBlock(dims[0], dp_rates[cum_sum[0]+_], norm_cfg=norm_cfg, widx=_) for _ in range(blocks[1])]),
        )

    def forward(self, x):
        c3, c4, c5 = x
        c4 = self.decoder[0](self.decoder_norm[0](self.lateral[0](c4) + self.decoder_upsample[0](c5)))
        c3 = self.decoder[1](self.decoder_norm[1](self.lateral[1](c3) + self.decoder_upsample[1](c4)))
        return c3, c4, c5


class CEDNet(nn.Module):

    def __init__(self,
                 in_chans=3,
                 num_stages=3,
                 encoder_type='Encoder',
                 decoder_type='Decoder',
                 base_block='NeXtBlock',
                 focal_block='DNeXtBlock',
                 p2_dim=96,
                 p2_block=3,
                 dims=[192, 352, 512],
                 blocks=[2, 4, 2],
                 layer_scale_init_value=1e-6,
                 drop_path_rate=0.1,
                 norm_cfg=dict(type='LN', eps=1e-6, data_format="channels_first"),
                 act_type='gelu',
                 num_classes=1000,
                 head_init_scale=1.0,
                 **kwargs):
        super().__init__()

        NORM_LAYERS.register_module('LN', force=True, module=LayerNorm)
        global BaseBlock, FocalBlock, act_layer, ls_init_value
        act_layer = {'gelu': nn.GELU, 'relu': nn.ReLU}.get(act_type, None)
        BaseBlock = globals().get(base_block, None)
        FocalBlock = globals().get(focal_block, None)
        ls_init_value = layer_scale_init_value

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 32, 3, 2, 1),
            build_norm_layer(norm_cfg, 32)[1],
            act_layer(),
            nn.Conv2d(32, p2_dim, 3, 2, 1),
            build_norm_layer(norm_cfg, p2_dim)[1],
            act_layer(),
        )

        if isinstance(blocks[0], int):
            blocks = [blocks for _ in range(num_stages)]

        max_num_blocks = p2_block + np.array(blocks).sum()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, max_num_blocks)]

        self.stages = nn.ModuleList()

        # p2 stage
        p2_stage = [BaseBlock(p2_dim, drop_path=dp_rates[_], norm_cfg=norm_cfg, widx=_) for _ in range(p2_block)]
        p2_stage.append(nn.Sequential(
            nn.Conv2d(p2_dim, dims[0], kernel_size=2, stride=2),
            build_norm_layer(norm_cfg, dims[0])[1]))
        self.stages.append(nn.Sequential(*p2_stage))

        # CEDNet stages
        for sidx in range(num_stages):
            sta_idx = p2_block + np.array(blocks[:sidx], dtype=np.int32).sum()
            end_dix = p2_block + np.array(blocks[:sidx+1], dtype=np.int32).sum()

            rates = dp_rates[sta_idx:end_dix]
            encoder_blocks = [blocks[sidx][0] // 2, blocks[sidx][1] // 2, blocks[sidx][2]]
            decoder_blocks = [blocks[sidx][1] - blocks[sidx][1] // 2, blocks[sidx][0] - blocks[sidx][0] // 2]
            if decoder_type in ['Decoder', 'DecoderFPN']:
                encoder_blocks = blocks[sidx]

            stage = [globals()[encoder_type](
                dims=dims, blocks=encoder_blocks, dp_rates=rates[:sum(encoder_blocks)], norm_cfg=norm_cfg)]
            if sidx < num_stages - 1:
                stage.append(globals()[decoder_type](
                    dims=dims, blocks=decoder_blocks, dp_rates=rates[sum(encoder_blocks):], norm_cfg=norm_cfg))
            self.stages.append(nn.Sequential(*stage))

        self.final_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.heads = nn.ModuleList()
        for _ in range(num_stages):
            self.final_layers.append(
                nn.Sequential(
                    nn.Conv2d(dims[-1], 2048, kernel_size=1, stride=1, padding=0),
                    build_norm_layer(norm_cfg, 2048)[1],
                    act_layer(),
                )
            )
            self.norms.append(nn.LayerNorm(2048, eps=1e-6))
            self.heads.append(nn.Linear(2048, num_classes))

        self.apply(self._init_weights)
        for head in self.heads:
            head.weight.data.mul_(head_init_scale)
            head.bias.data.mul_(head_init_scale)

        NORM_LAYERS.register_module('LN', force=True, module=nn.LayerNorm)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_training(self, x):
        x = self.stem(x)
        x = self.stages[0](x)
        stage_feats = []
        for stage in self.stages[1:]:
            x = stage(x)
            stage_feats.append(x)

        outs = []
        for i, feats in enumerate(stage_feats):
            out = self.final_layers[i](feats[-1])
            out = self.heads[i](self.norms[i](out.mean([-2, -1])))
            outs.append(out)
        return outs

    def forward_inference(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        out = self.final_layers[-1](x[-1])
        out = self.heads[-1](self.norms[-1](out.mean([-2, -1])))
        return out

    def forward(self, x):
        return self.forward_training(x) if self.training else self.forward_inference(x)


@register_model
def CEDNet_next_tiny(**kwargs): # drop_path_rate = 0.1
    model = CEDNet(num_stages=3, p2_dim=96, p2_block=3, dims=[192, 352, 512], blocks=[2, 4, 2], **kwargs)
    return model


@register_model
def CEDNet_next_small(**kwargs): # drop_path_rate = 0.4
    model = CEDNet(num_stages=4, p2_dim=96, p2_block=3, dims=[192, 352, 512], blocks=[2, 7, 2], **kwargs)
    return model


@register_model
def CEDNet_next_base(**kwargs): # drop_path_rate = 0.5
    model = CEDNet(num_stages=4, p2_dim=128, p2_block=3, dims=[256, 448, 704], blocks=[2, 7, 2], **kwargs)
    return model


@register_model
def CEDNet_swin_tiny(**kwargs): # drop_path_rate = 0.1
    model = CEDNet(
        base_block='SwinBlock', focal_block='DSwinBlock',
        num_stages=3, p2_dim=96, p2_block=3, dims=[160, 288, 416], blocks=[2, 4, 2],
        **kwargs)
    return model


@register_model
def CEDNet_res50(**kwargs): # drop_path_rate = 0.
    model = CEDNet(
        base_block='Bottleneck', focal_block='DBottleneck',
        num_stages=3, p2_dim=64, p2_block=3, dims=[128, 256, 384], blocks=[1, 4, 2],
        norm_cfg=dict(type='BN', requires_grad=True), act_type='relu', **kwargs)
    return model


@register_model
def CEDNet_next_tiny_fpn(**kwargs):
    model = CEDNet(
        encoder_type='EncoderFPN',
        decoder_type='DecoderFPN',
        num_stages=3, p2_dim=96, p2_block=3, dims=[160, 304, 448], blocks=[2, 4, 2], **kwargs)
    return model


@register_model
def CEDNet_next_tiny_unet(**kwargs):
    model = CEDNet(
        decoder_type='DecoderUNet',
        num_stages=3, p2_dim=96, p2_block=3, dims=[192, 352, 512], blocks=[2, 4, 2], **kwargs)
    return model


@register_model
def CEDNet_next_tiny_hourglass(**kwargs):
    model = CEDNet(
        decoder_type='DecoderHourglass',
        num_stages=3, p2_dim=96, p2_block=3, dims=[160, 320, 512], blocks=[2, 4, 2], **kwargs)
    return model


if __name__ == "__main__":
    from mmcv.cnn.utils import get_model_complexity_info

    model = CEDNet_next_tiny()
    # model = CEDNet_next_tiny_fpn()
    # model = CEDNet_next_tiny_unet()
    # model = CEDNet_next_tiny_hourglass()

    input_shape = (3, 224, 224)
    flops, params = get_model_complexity_info(model, input_shape)

    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\nFlops: {flops}\nParams: {params}\n{split_line}')
