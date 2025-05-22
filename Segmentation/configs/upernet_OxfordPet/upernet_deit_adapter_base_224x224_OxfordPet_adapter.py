# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/upernet_r50.py',  '../_base_/datasets/Oxford-IIIT_adapter.py',
    '../_base_/default_runtime.py', '../_base_/schedules/epoch.py'
]
pretrained = 'media/dl_shouan/ZHITAI/Adapter_test/mmseg_custom/pre_weight/jx_vit_base_p16_224-80ecf9dd.pth'
model = dict(
    pretrained=pretrained,
    backbone=dict(
        _delete_=True,
        type='ViTAdapter',
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_rate=0.3,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        window_attn=[False] * 12,
        window_size=[None] * 12),
    decode_head=dict(num_classes=3, in_channels=[768, 768, 768, 768]),
    auxiliary_head=dict(num_classes=3, in_channels=768),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=4)
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)