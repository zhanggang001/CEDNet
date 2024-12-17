_base_ = ['./deformable_detr_r50_16x2_50e_coco.py']

model = dict(
    type='DeformableDETR',
    backbone=dict(
        _delete_=True,
        type='CEDNetWrapper',
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
        out_chans=256,
        num_add_blocks=1,
        layer_scale_init_value=1e-6,
        drop_path_rate=0.2,
        norm_cfg=dict(type='LN', eps=1e-6, data_format="channels_first"),
        act_type='gelu',
    ),
    neck=None,
    bbox_head=dict(transformer=dict(encoder=dict(num_layers=6)))
)


optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW',
                 lr=0.0002, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.5,
                                'decay_type': 'layer_wise_cednet',
                                'num_layers': 4})
