_base_ = ['../convnext/retinanet_convnext_tiny_patch4_window7_copy_paste_lsj_adamw_3x_coco_in1k.py']

model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='CEDNetWrapper',
        in_chans=3,
        num_stages=3,
        encoder_type='Encoder',
        decoder_type='Decoder',
        base_block='Bottleneck',
        focal_block='DBottleneck',
        p2_dim=64,
        p2_block=3,
        dims=[128, 256, 384],
        blocks=[1, 4, 2],
        out_chans=256,
        num_add_blocks=2,
        layer_scale_init_value=1e-6,
        drop_path_rate=0.,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_type='relu',
    ),
    neck=None,
)

optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW',
                 lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 1.0,
                                'decay_type': 'layer_wise_cednet',
                                'num_layers': 4})
