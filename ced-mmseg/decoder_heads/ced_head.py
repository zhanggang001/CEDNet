import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.decode_heads.psp_head import PPM


@HEADS.register_module()
class CEDHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        super(CEDHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        # self.head_convs = nn.ModuleList()
        # for channels in self.in_channels:
        #     h_conv = ConvModule(
        #         channels,
        #         channels,
        #         3,
        #         conv_cfg=self.conv_cfg,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg,
        #         inplace=False)
        #     self.head_convs.append(h_conv)

        self.head_bottleneck = ConvModule(
            sum(self.in_channels),
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        backbone_feats = self._transform_inputs(inputs)
        # for i, head_conv in enumerate(self.head_convs):
        #     backbone_feats[i] = head_conv(backbone_feats[i])

        backbone_feats_levels = len(backbone_feats)
        for i in range(backbone_feats_levels - 1, 0, -1):
            backbone_feats[i] = resize(
                backbone_feats[i],
                size=backbone_feats[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        backbone_feats = torch.cat(backbone_feats, dim=1)
        feats = self.head_bottleneck(backbone_feats)
        return feats

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
