_base_ = ['../convnext/retinanet_convnext_tiny_patch4_window7_copy_paste_lsj_adamw_3x_coco_in1k.py']

norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    pretrained=None,
    backbone=dict(norm_cfg=norm_cfg),
    neck=dict(
        type='BiFPNScratch',
        input_channels=[96, 192, 384, 768],
        dims=[256, 256, 256],
        blocks=[1, 1, 1],
        num_layers=3,
        norm_cfg=norm_cfg,
    ),
    bbox_head=dict(norm_cfg=norm_cfg)
)

optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW',
                 lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 1.0,
                                'decay_type': 'layer_wise',
                                'num_layers': 6})
