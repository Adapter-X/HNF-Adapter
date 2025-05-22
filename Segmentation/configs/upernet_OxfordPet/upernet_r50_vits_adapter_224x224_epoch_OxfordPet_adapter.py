_base_ = [
    '../_base_/models/upernet_r50_vitb.py', '../_base_/datasets/Oxford-IIIT_adapter.py',
    '../_base_/default_runtime.py', '../_base_/schedules/epoch.py'
]

model = dict(
    backbone=dict(type='Resnet50_ViT_B',
                  f=76, h=76, m=16, l=64, r=0.1, s=0.1, a=0.1, b=1, img_size=224),
    decode_head=dict(in_channels=[512, 512, 1024, 2048], num_classes=3),
    auxiliary_head=dict(in_channels=1024, num_classes=3))
# runner = dict(type='IterBasedRunner')
# checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
# evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU', pre_eval=True)
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

# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)


