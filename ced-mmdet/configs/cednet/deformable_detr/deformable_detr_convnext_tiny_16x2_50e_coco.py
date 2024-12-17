_base_ = ['./deformable_detr_r50_16x2_50e_coco.py']

model = dict(
    type='DeformableDETR',
    backbone=dict(
        _delete_=True,
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[1, 2, 3],
    ),
    neck=dict(in_channels=[192, 384, 768]),
)
