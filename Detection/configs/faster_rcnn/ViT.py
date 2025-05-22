_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc12_adapter.py',
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]
pretrained = 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='ViTBaseline',
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_rate=0.1,
        out_indices=[2, 5, 8, 11],
        # window_attn=[True, True, False, True, True, False,
        #              True, True, False, True, True, False],
        # window_size=[14, 14, None, 14, 14, None,
        #              14, 14, None, 14, 14, None],
        pretrained=pretrained),
    neck=dict(
        type='FPN',
        in_channels=[768, 768, 768, 768],
        out_channels=256,
        num_outs=5))
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