_base_ = ['cascade_mask_rcnn_convnext_small_patch4_window7_copy_paste_lsj_giou_4conv1f_adamw_3x_coco_in1k.py']

model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='CEDNetWrapperTwoStage',
        in_chans=3,
        num_stages=4,
        encoder_type='Encoder',
        decoder_type='Decoder',
        base_block='NeXtBlock',
        focal_block='DNeXtBlock',
        p2_dim=128,
        p2_block=3,
        dims=[256, 448, 704],
        blocks=[2, 7, 2],
        out_chans=256,
        num_add_blocks=1,
        layer_scale_init_value=1e-6,
        drop_path_rate=0.4,
        norm_cfg=dict(type='LN', eps=1e-6, data_format="channels_first"),
        act_type='gelu',
    ),
    neck=None,
)

optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW',
                 lr=0.0002, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.7,
                                'decay_type': 'layer_wise_cednet_small',
                                'num_layers': 9})
