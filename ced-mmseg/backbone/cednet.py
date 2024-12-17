import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from mmcv.cnn import build_norm_layer, ConvModule
from mmcv.cnn.bricks.registry import NORM_LAYERS

from mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES
from mmseg.models.decode_heads.psp_head import PPM


act_layer = nn.GELU
BaseBlock = None
FocalBlock = None
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


class LastDecoder(nn.Module):

    def __init__(self, dims=[96, 192, 352, 512], norm_cfg=None, **kwargs):
        super().__init__()

        self.decoder_upsample = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dims[3], dims[2], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)),
            nn.Sequential(nn.Conv2d(dims[2], dims[1], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)),
            nn.Sequential(nn.Conv2d(dims[1], dims[0], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        ])

        self.decoder_norm = nn.Sequential(
            build_norm_layer(norm_cfg, dims[2])[1],
            build_norm_layer(norm_cfg, dims[1])[1],
            build_norm_layer(norm_cfg, dims[0])[1],
        )

    def forward(self, x):
        c2, c3, c4, c5 = x
        c4 = self.decoder_norm[0](c4 + self.decoder_upsample[0](c5))
        c3 = self.decoder_norm[1](c3 + self.decoder_upsample[1](c4))
        c2 = self.decoder_norm[2](c2 + self.decoder_upsample[2](c3))
        return c2, c3, c4, c5


@BACKBONES.register_module()
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
                 last_decoder=False,
                 layer_scale_init_value=1e-6,
                 drop_path_rate=0.1,
                 norm_cfg=dict(type='LN', eps=1e-6, data_format="channels_first"),
                 act_type='gelu',
                 pretrained=None):
        super().__init__()

        NORM_LAYERS.register_module('LN', force=True, module=LayerNorm)

        global act_layer, BaseBlock, FocalBlock, ls_init_value
        act_layer = {'gelu': nn.GELU, 'relu': nn.ReLU}.get(act_type, None)
        BaseBlock = globals().get(base_block, None)
        FocalBlock = globals().get(focal_block, None)
        ls_init_value = layer_scale_init_value

        self.num_stages = num_stages
        self.dims = dims
        self.norm_cfg = norm_cfg
        self.act_type = act_type
        self.pretrained = pretrained

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
            stage = [globals()[encoder_type](
                dims=dims, blocks=blocks[sidx], dp_rates=dp_rates[sta_idx:end_dix], norm_cfg=norm_cfg)]
            if sidx < num_stages - 1 or last_decoder:
                stage.append(globals()[decoder_type](dims=dims, norm_cfg=norm_cfg))
            self.stages.append(nn.Sequential(*stage))

        self.init_weights()
        NORM_LAYERS.register_module('LN', force=True, module=nn.LayerNorm)

    def init_weights(self):

        def _init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight.shape[0] > 1:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.stages[0][:-1](x)
        x = self.stages[0][-1](c2)

        x_hist = []
        for stage in self.stages[1:]:
            x = stage(x)
            x_hist.append(x)
        x_hist = x_hist[:-1][::-1]

        return [c2] + list(x) + [out[0] for out in x_hist]


@BACKBONES.register_module()
class CEDNetWrapper(CEDNet):

    def __init__(self, out_channels=512, pool_scales=(1, 2, 3, 6), extra_norm_cfg=None, extra_act_cfg=None, *args, **kwargs):
        super(CEDNetWrapper, self).__init__(*args, **kwargs)

        dims = [min(dim, out_channels) for dim in kwargs['dims']]
        drop_path_rate = kwargs['drop_path_rate']
        blocks = kwargs['blocks']

        self.reduce_convs = nn.ModuleList()
        for input_dim, outpu_dim in zip(kwargs['dims'], dims):
            reduce_conv = nn.Sequential(
                nn.Conv2d(input_dim, outpu_dim, 1),
                build_norm_layer(extra_norm_cfg, outpu_dim)[1])
            self.reduce_convs.append(reduce_conv)

        self.last_encoder = Encoder(
            dims=dims, blocks=blocks, dp_rates=[drop_path_rate] * sum(blocks), norm_cfg=extra_norm_cfg)

        self.psp_modules = PPM(
            pool_scales,
            dims[-1],
            out_channels,
            conv_cfg=None,
            norm_cfg=extra_norm_cfg,
            act_cfg=extra_act_cfg,
            align_corners=False)

        self.bottleneck = ConvModule(
            dims[-1] + len(pool_scales) * out_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=None,
            norm_cfg=extra_norm_cfg,
            act_cfg=extra_act_cfg)

        dims = [kwargs['p2_dim'], dims[0], dims[1], out_channels]
        self.last_decoder = LastDecoder(dims, extra_norm_cfg)
        self.init_weights()

    def init_weights(self, pretrained=None):

        def _init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight.shape[0] > 1:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

    def forward(self, x):
        ret_feats = super().forward(x)
        pre_c2, pre_c3, pre_c4, pre_c5 = ret_feats[:4]
        c3, c4, c5 = self.last_encoder((pre_c3, pre_c4, pre_c5))

        c3 = self.reduce_convs[0](c3)
        c4 = self.reduce_convs[1](c4)
        c5 = self.reduce_convs[2](c5)

        psp_outs = [c5]
        psp_outs.extend(self.psp_modules(c5))
        psp_outs = torch.cat(psp_outs, dim=1)
        c5 = self.bottleneck(psp_outs)

        c2, c3, c4, c5 = self.last_decoder((pre_c2, c3, c4, c5))
        return [c2, c3, c4, c5, pre_c3] + ret_feats[4:]
