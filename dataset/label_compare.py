"""
Label Comparison Dataset
用于对比两种模型生成标签的差异

输入splits文件格式: image_path old_mask_path new_mask_path
"""

import os
import math
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class LabelCompareDataset(Dataset):
    """
    对比两种标签的数据集 (参考 SemiDataset 实现)

    splits文件格式:
        image_path old_mask_path new_mask_path

    Args:
        name: 数据集名称 (如 loveda, potsdam 等)
        root: 数据集根目录
        mode: 模式 ('compare')
        id_path: splits文件路径
        ignore_value: 忽略的标签值 (默认 255)
    """

    def __init__(
        self,
        name,
        root,
        mode="compare",
        ignore_value=255,
    ):
        self.name = name
        self.root = root
        self.mode = mode
        self.ignore_value = ignore_value

        # 读取splits文件
        with open("splits/%s/compare.txt" % name, "r") as f:
            self.ids = f.read().splitlines()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        """
        Returns:
            img_path: 图像路径
            old_mask: 原标签 (numpy array)
            new_mask: 新标签 (numpy array)
            id: 原始行内容
        """
        id = self.ids[item]

        img_path = id.split(" ")[0]
        old_mask_path = id.split(" ")[1]
        new_mask_path = id.split(" ")[2]

        # 加载标签
        old_mask = Image.open(os.path.join(self.root, old_mask_path))
        new_mask = Image.open(os.path.join(self.root, new_mask_path))

        # 处理特殊数据集 (参考 SemiDataset)
        if self.name == "loveda":
            old_mask = self.process_mask(old_mask)
            new_mask = self.process_mask(new_mask)

        old_mask = np.array(old_mask)
        new_mask = np.array(new_mask)

        return img_path, old_mask, new_mask, id

    def process_mask(self, mask):
        """处理 loveda 数据集的 mask (参考 SemiDataset)"""
        mask = np.array(mask) - 1
        return Image.fromarray(mask)

    def compute_diff_ratio(self, old_mask, new_mask):
        """
        计算两个标签的像素不一致率

        Args:
            old_mask: 原标签 (numpy array)
            new_mask: 新标签 (numpy array)

        Returns:
            float: 不一致率 (0.0 ~ 1.0)
        """
        # 创建有效像素掩码（排除ignore区域）
        valid_mask = (old_mask != self.ignore_value) & (new_mask != self.ignore_value)

        if valid_mask.sum() == 0:
            return 0.0

        # 计算差异
        diff = (old_mask != new_mask) & valid_mask
        return float(diff.sum()) / float(valid_mask.sum())
