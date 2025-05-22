_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/Oxford-IIIT_adapter.py',
    '../_base_/default_runtime.py', '../_base_/schedules/epoch.py'
]
model = dict(
    decode_head=dict(num_classes=3), auxiliary_head=dict(num_classes=3))
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=4)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)
