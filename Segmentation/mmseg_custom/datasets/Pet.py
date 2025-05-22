# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class Oxford_IIIT_PetDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('Background', 'Foreground', 'Not classified')

    PALETTE = [[0, 0, 0], [255, 255, 255], [255, 0, 0]]

    def __init__(self, split, **kwargs):
        super(Oxford_IIIT_PetDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None