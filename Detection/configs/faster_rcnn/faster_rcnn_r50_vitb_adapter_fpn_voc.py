# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/faster_rcnn_Resnet50_ViTB.py',
    '../_base_/datasets/voc12_adapter.py',
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(type='Resnet50_ViT_B',
                  f=76, h=76, m=16, l=64, r=0.1, s=0.1, a=0.1, b=1, num_classes=20),
    neck=dict(
        type='FPN',
        in_channels=[512, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    roi_head=dict(bbox_head=dict(num_classes=20)))
# optimizer
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# augmentation strategy originates from DETR / Sparse RCNN
optimizer = dict(_delete_=True,
                 type='AdamW',
                 lr=0.0001,
                 weight_decay=0.05,
                 # paramwise_cfg=dict(
                 #     custom_keys={
                 #         'level_embed': dict(decay_mult=0.),
                 #         'pos_embed': dict(decay_mult=0.),
                 #         'norm': dict(decay_mult=0.),
                 #         'bias': dict(decay_mult=0.)
                 #     })
                 )
optimizer_config = dict(grad_clip=None)
fp16 = dict(loss_scale=dict(init_scale=512))
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=3,
    save_last=True,
)