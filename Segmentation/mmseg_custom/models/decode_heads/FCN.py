import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torchvision.models
from mmseg.models.builder import HEADS

class FCNHead(nn.Module):
    """完全卷积网络用于语义分割。

    该头部是基于FCNNet <https://arxiv.org/abs/1411.4038>_实现的。

    Args:
        num_convs (int): 头部中的卷积层数。默认值: 2。
        kernel_size (int): 头部中卷积的卷积核大小。默认值: 3。
        concat_input (bool): 是否在分类层之前将输入和卷积的输出进行连接。
        dilation (int): 头部中卷积的膨胀率。默认值: 1。
    """

    def __init__(self, num_convs=2, kernel_size=3, concat_input=True, dilation=1):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super().__init__()
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        """用于在使用self.cls_seg全连接层对每个像素进行分类之前对特征图进行前向传播的函数。

        参数:
        inputs (list[Tensor]): 多级图像特征的列表。

        返回值:
        feats (Tensor): 形状为(batch_size, self.channels, H, W)的张量，是解码器最后一层的特征图。
        """
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
