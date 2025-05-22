# Copyright (c) OpenMMLab. All rights reserved.
# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[{
            'type':
            'Resize',
            'img_scale': [(256, 700), (288, 700), (320, 700), (352, 700),
                          (384, 700), (416, 700), (448, 700), (480, 700),
                          (512, 700), (544, 700)],
            'multiscale_mode':
            'value',
            'keep_ratio':
            True
        }],
                  [{
                      'type': 'Resize',
                      'img_scale': [(200, 700), (300, 700), (400, 700)],
                      'multiscale_mode': 'value',
                      'keep_ratio': True
                  }, {
                      'type': 'RandomCrop',
                      'crop_type': 'absolute_range',
                      'crop_size': (224, 400),
                      'allow_negative_crop': True
                  }, {
                      'type':
                      'Resize',
                      'img_scale': [(256, 700), (288, 700), (320, 700),
                                    (352, 700), (384, 700), (416, 700),
                                    (448, 700), (480, 700), (512, 700),
                                    (544, 700)],
                      'multiscale_mode':
                      'value',
                      'override':
                      True,
                      'keep_ratio':
                      True
                  }]]),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(500, 500),
        allow_negative_crop=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 400),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(type='RepeatDataset',
               times=3,
               dataset=dict(
                   type=dataset_type,
                   ann_file=data_root + 'VOC2012/ImageSets/train_images.txt',
                   img_prefix=data_root + 'VOC2012/',
                   pipeline=train_pipeline)),
    val=dict(type=dataset_type,
             ann_file=data_root + 'VOC2012/ImageSets/test_images.txt',
             img_prefix=data_root + 'VOC2012/',
             pipeline=test_pipeline),
    test=dict(type=dataset_type,
              ann_file=data_root + 'VOC2012/ImageSets/test_images.txt',
              img_prefix=data_root + 'VOC2012/',
              pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
