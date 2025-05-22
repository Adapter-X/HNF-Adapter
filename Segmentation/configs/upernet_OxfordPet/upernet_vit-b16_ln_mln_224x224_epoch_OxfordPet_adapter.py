_base_ = [
    '../_base_/models/upernet_vit-b16_ln_mln.py',
    '../_base_/datasets/Oxford-IIIT_adapter.py', '../_base_/default_runtime.py',
    '../_base_/schedules/epoch.py'
]

model = dict(
    pretrained='/media/dl_shouan/ZHITAI/Adapter_test/mmseg_custom/pre_weight/jx_vit_base_p16_224-80ecf9dd.pth',
    backbone=dict(drop_path_rate=0.1, final_norm=True, img_size=(224, 224)),
    decode_head=dict(num_classes=3),
    auxiliary_head=dict(num_classes=3))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=12)