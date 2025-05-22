_base_ = [
    '../_base_/models/upernet_Adapter.py',
    '../_base_/datasets/Oxford-IIIT_adapter.py', '../_base_/default_runtime.py',
    '../_base_/schedules/epoch.py'
]

model = dict(
    decode_head=dict(num_classes=3),
    auxiliary_head=dict(num_classes=3))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
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