# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=1000)
#共1000个epoch
checkpoint_config = dict(by_epoch=True, interval=100)
#每一次epoch都保存一次模型
evaluation = dict(interval=40, metric='mIoU', pre_eval=True, save_best='mIoU')
#每一次epoch都计算一次验证