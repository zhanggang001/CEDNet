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
        base_block='SwinBlock',
        focal_block='DSwinBlock',
        p2_dim=96,
        p2_block=3,
        dims=[160, 288, 416],
        blocks=[2, 4, 2],
        out_chans=256,
        num_add_blocks=2,
        layer_scale_init_value=1e-6,
        drop_path_rate=0.1,
        norm_cfg=dict(type='LN', eps=1e-6, data_format="channels_first"),
        act_type='gelu',
    ),
    neck=None,
)

optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    _delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.95,
        'decay_type': 'layer_wise_cednet',
        'num_layers': 4,
        'custom_keys': {
            'absolute_pos_embed': {'decay_mult':0.},
            'relative_position_bias_table': {'decay_mult':0.},
            'norm': {'decay_mult': 0.}
        }
    }
)
