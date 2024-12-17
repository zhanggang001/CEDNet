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
        base_block='Bottleneck',
        focal_block='DBottleneck',
        p2_dim=64,
        p2_block=3,
        dims=[128, 256, 384],
        blocks=[1, 4, 2],
        out_chans=256,
        num_add_blocks=1,
        layer_scale_init_value=1e-6,
        drop_path_rate=0.,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_type='relu',
    ),
    neck=None,
    bbox_head=dict(transformer=dict(encoder=dict(num_layers=6)))
)
