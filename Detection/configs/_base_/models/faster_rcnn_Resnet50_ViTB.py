# 模型设置
model = dict(
    type='FasterRCNN',  # 模型类型为Faster R-CNN
    backbone=dict(type='Resnet50_ViT_B',  # 主干网络类型为Resnet50_ViT_B
                  f=76, h=76, m=16, l=64, r=0.1, s=0.1, a=0.1, b=1, num_classes=20),  # 主干网络的参数配置
    neck=dict(
        type='FPN',  # 特征金字塔网络（FPN）类型
        in_channels=[768, 768, 768, 768],  # 每层输入的通道数
        out_channels=256,  # 输出通道数
        num_outs=5),  # 输出层数
    rpn_head=dict(
        type='RPNHead',  # 区域建议网络头部（RPNHead）类型
        in_channels=256,  # 输入通道数
        feat_channels=256,  # 特征通道数
        anchor_generator=dict(
            type='AnchorGenerator',  # 锚框生成器类型
            scales=[8],  # 锚框的缩放因子
            ratios=[0.5, 1.0, 2.0],  # 锚框的宽高比
            strides=[4, 8, 16, 32, 64]),  # 锚框的步幅
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',  # 边界框编码器类型
            target_means=[.0, .0, .0, .0],  # 边界框目标均值
            target_stds=[1.0, 1.0, 1.0, 1.0]),  # 边界框目标标准差
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),  # 分类损失函数类型和权重
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),  # 边界框回归损失函数类型和权重
    roi_head=dict(
        type='StandardRoIHead',  # RoI头部类型
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',  # 单一RoI提取器类型
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),  # RoI对齐层的配置
            out_channels=256,  # 输出通道数
            featmap_strides=[4, 8, 16, 32]),  # 特征图步幅
        bbox_head=dict(
            type='Shared2FCBBoxHead',  # 共享全连接边界框头部类型
            in_channels=256,  # 输入通道数
            fc_out_channels=1024,  # 全连接层输出通道数
            roi_feat_size=7,  # RoI特征图大小
            num_classes=80,  # 类别数
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',  # 边界框编码器类型
                target_means=[0., 0., 0., 0.],  # 边界框目标均值
                target_stds=[0.1, 0.1, 0.2, 0.2]),  # 边界框目标标准差
            reg_class_agnostic=False,  # 是否不考虑类别的回归
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),  # 分类损失函数类型和权重
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),  # 边界框回归损失函数类型和权重
    # 模型训练和测试设置
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',  # 最大IoU分配器类型
                pos_iou_thr=0.7,  # 正样本的IoU阈值
                neg_iou_thr=0.3,  # 负样本的IoU阈值
                min_pos_iou=0.3,  # 最小正样本IoU
                match_low_quality=True,  # 是否匹配低质量的样本
                ignore_iof_thr=-1),  # 忽略的IoF阈值
            sampler=dict(
                type='RandomSampler',  # 随机采样器类型
                num=256,  # 采样数量
                pos_fraction=0.5,  # 正样本比例
                neg_pos_ub=-1,  # 负样本上限
                add_gt_as_proposals=False),  # 是否将真实目标作为建议框
            allowed_border=-1,  # 允许的边界
            pos_weight=-1,  # 正样本权重
            debug=False),  # 调试模式
        rpn_proposal=dict(
            nms_pre=2000,  # NMS前的候选框数
            max_per_img=1000,  # 每张图片的最大建议框数
            nms=dict(type='nms', iou_threshold=0.7),  # NMS类型和IoU阈值
            min_bbox_size=0),  # 最小边界框大小
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',  # 最大IoU分配器类型
                pos_iou_thr=0.5,  # 正样本的IoU阈值
                neg_iou_thr=0.5,  # 负样本的IoU阈值
                min_pos_iou=0.5,  # 最小正样本IoU
                match_low_quality=False,  # 是否匹配低质量的样本
                ignore_iof_thr=-1),  # 忽略的IoF阈值
            sampler=dict(
                type='RandomSampler',  # 随机采样器类型
                num=512,  # 采样数量
                pos_fraction=0.25,  # 正样本比例
                neg_pos_ub=-1,  # 负样本上限
                add_gt_as_proposals=True),  # 是否将真实目标作为建议框
            pos_weight=-1,  # 正样本权重
            debug=False)),  # 调试模式
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,  # NMS前的候选框数
            max_per_img=1000,  # 每张图片的最大建议框数
            nms=dict(type='nms', iou_threshold=0.7),  # NMS类型和IoU阈值
            min_bbox_size=0),  # 最小边界框大小
        rcnn=dict(
            score_thr=0.05,  # 预测分数阈值
            nms=dict(type='nms', iou_threshold=0.5),  # NMS类型和IoU阈值
            max_per_img=100)  # 每张图片的最大检测框数
        # soft-nms也支持用于rcnn测试
        # 例如，nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
