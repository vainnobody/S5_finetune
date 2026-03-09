"""
错误区域检测模块

实现多种错误检测方法:
1. 预测-标签不一致性检测
2. 低置信度区域检测
3. 多尺度错误感知定位模块 (Ms-EALM)
4. 综合错误检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class ErrorDetector:
    """
    错误区域检测器
    
    用于识别标签中可能存在错误的区域。
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.8,
        entropy_threshold: float = 2.0,
        inconsistency_weight: float = 0.5,
        ignore_index: int = 255
    ):
        """
        Args:
            confidence_threshold: 置信度阈值
            entropy_threshold: 熵阈值
            inconsistency_weight: 不一致性权重
            ignore_index: 忽略的标签值
        """
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.inconsistency_weight = inconsistency_weight
        self.ignore_index = ignore_index
    
    def detect(
        self,
        prediction: torch.Tensor,
        label: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        method: str = 'comprehensive'
    ) -> torch.Tensor:
        """
        检测错误区域
        
        Args:
            prediction: 预测概率 (B, C, H, W)
            label: 标签 (B, H, W)
            confidence: 置信度图 (B, 1, H, W) (可选)
            uncertainty: 不确定性图 (B, 1, H, W) (可选)
            method: 检测方法 ('inconsistency', 'confidence', 'comprehensive')
            
        Returns:
            error_mask: 错误区域掩码 (B, 1, H, W)
        """
        if method == 'inconsistency':
            return self.detect_inconsistency(prediction, label)
        elif method == 'confidence':
            return self.detect_low_confidence(confidence)
        elif method == 'comprehensive':
            return self.comprehensive_detection(
                prediction, label, confidence, uncertainty
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def detect_inconsistency(
        self,
        prediction: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        检测预测与标签不一致的区域
        
        当模型预测与标签不同时，可能是标签错误。
        
        Args:
            prediction: 预测概率 (B, C, H, W)
            label: 标签 (B, H, W)
            
        Returns:
            inconsistency_mask: 不一致区域掩码 (B, 1, H, W)
        """
        # 获取预测类别
        pred_class = prediction.argmax(dim=1)  # (B, H, W)
        
        # 有效区域掩码
        valid_mask = (label != self.ignore_index)
        
        # 不一致区域
        inconsistency = (pred_class != label) & valid_mask
        
        return inconsistency.unsqueeze(1).float()  # (B, 1, H, W)
    
    def detect_low_confidence(
        self,
        confidence: torch.Tensor
    ) -> torch.Tensor:
        """
        检测低置信度区域
        
        低置信度区域可能是标签错误或模糊区域。
        
        Args:
            confidence: 置信度图 (B, 1, H, W)
            
        Returns:
            low_conf_mask: 低置信度区域掩码 (B, 1, H, W)
        """
        low_conf = confidence < self.confidence_threshold
        return low_conf.float()
    
    def detect_high_entropy(
        self,
        entropy: torch.Tensor
    ) -> torch.Tensor:
        """
        检测高熵（高不确定性）区域
        
        Args:
            entropy: 熵图 (B, 1, H, W)
            
        Returns:
            high_entropy_mask: 高熵区域掩码 (B, 1, H, W)
        """
        high_entropy = entropy > self.entropy_threshold
        return high_entropy.float()
    
    def comprehensive_detection(
        self,
        prediction: torch.Tensor,
        label: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        综合错误检测
        
        结合多种指标进行错误检测：
        1. 预测-标签不一致
        2. 高置信度不一致（模型非常确定预测与标签不同）
        3. 低置信度区域
        
        Args:
            prediction: 预测概率 (B, C, H, W)
            label: 标签 (B, H, W)
            confidence: 置信度图 (B, 1, H, W)
            uncertainty: 不确定性图 (B, 1, H, W)
            
        Returns:
            error_mask: 错误区域掩码 (B, 1, H, W)
        """
        # 获取预测类别
        pred_class = prediction.argmax(dim=1)  # (B, H, W)
        max_prob = prediction.max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
        
        # 有效区域掩码
        valid_mask = (label != self.ignore_index).unsqueeze(1)  # (B, 1, H, W)
        
        # 预测与标签不一致
        inconsistency = (pred_class != label).unsqueeze(1).float()  # (B, 1, H, W)
        
        # 高置信度不一致 = 模型非常确定但与标签不同
        high_conf_inconsistency = (
            inconsistency * 
            (max_prob > self.confidence_threshold).float()
        )
        
        # 基础错误掩码
        error_mask = high_conf_inconsistency
        
        # 如果有额外的置信度信息
        if confidence is not None:
            # 低置信度区域可能是标签问题
            low_conf_error = (
                inconsistency * 
                (confidence < self.confidence_threshold).float() * 
                0.5  # 降低权重
            )
            error_mask = torch.max(error_mask, low_conf_error)
        
        # 应用有效区域掩码
        error_mask = error_mask * valid_mask.float()
        
        return error_mask
    
    def detect_boundary_errors(
        self,
        prediction: torch.Tensor,
        label: torch.Tensor,
        boundary_width: int = 3
    ) -> torch.Tensor:
        """
        检测边界区域的错误
        
        边界区域通常是标注最容易出错的地方。
        
        Args:
            prediction: 预测概率 (B, C, H, W)
            label: 标签 (B, H, W)
            boundary_width: 边界宽度
            
        Returns:
            boundary_error_mask: 边界错误掩码 (B, 1, H, W)
        """
        # 检测标签边界
        label_boundary = self.extract_boundary(label.unsqueeze(1), boundary_width)
        
        # 检测不一致
        inconsistency = self.detect_inconsistency(prediction, label)
        
        # 边界区域的不一致
        boundary_error = inconsistency * label_boundary
        
        return boundary_error
    
    @staticmethod
    def extract_boundary(
        mask: torch.Tensor,
        width: int = 3
    ) -> torch.Tensor:
        """
        提取边界区域
        
        Args:
            mask: 掩码 (B, 1, H, W)
            width: 边界宽度
            
        Returns:
            boundary: 边界掩码 (B, 1, H, W)
        """
        # 使用形态学操作提取边界
        kernel_size = 2 * width + 1
        
        # 膨胀
        dilated = F.max_pool2d(
            mask, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=width
        )
        
        # 腐蚀
        eroded = -F.max_pool2d(
            -mask, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=width
        )
        
        # 边界 = 膨胀 - 腐蚀
        boundary = (dilated - eroded).clamp(0, 1)
        
        return boundary


class MultiScaleErrorAwareLocalization(nn.Module):
    """
    多尺度错误感知定位模块 (Ms-EALM)
    
    参考: IUR-Net (Remote Sensing 2025)
    
    通过多尺度特征融合捕获图像与标签之间的不一致性，
    实现更精确的错误区域定位。
    """
    
    def __init__(
        self,
        in_channels: List[int],
        hidden_dim: int = 64,
        num_scales: int = 4,
        ignore_index: int = 255
    ):
        """
        Args:
            in_channels: 各尺度输入通道数列表
            hidden_dim: 隐藏层维度
            num_scales: 尺度数量
            ignore_index: 忽略的标签值
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.num_scales = num_scales
        
        # 多尺度特征处理
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            for in_ch in in_channels
        ])
        
        # 多尺度深度可分离卷积
        self.dw_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=i, 
                         dilation=i, groups=hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            for i in range(1, num_scales + 1)
        ])
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * num_scales, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        multi_scale_features: List[torch.Tensor],
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            multi_scale_features: 多尺度特征列表
            label: 标签 (B, H, W)
            
        Returns:
            error_prob: 错误概率图 (B, 1, H, W)
        """
        target_size = multi_scale_features[0].shape[2:]
        
        # 处理每个尺度的特征
        processed_features = []
        for i, (feat, scale_conv, dw_conv) in enumerate(
            zip(multi_scale_features, self.scale_convs, self.dw_convs)
        ):
            # 通道变换
            feat = scale_conv(feat)
            
            # 上采样到目标尺寸
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode='bilinear', align_corners=True
                )
            
            # 多尺度深度可分离卷积
            feat = dw_conv(feat)
            processed_features.append(feat)
        
        # 特征融合
        concat_feat = torch.cat(processed_features, dim=1)
        error_prob = self.fusion(concat_feat)
        
        return error_prob


class ErrorRegionAnalyzer:
    """
    错误区域分析器
    
    分析错误区域的统计特性和空间分布。
    """
    
    def __init__(self, ignore_index: int = 255):
        self.ignore_index = ignore_index
    
    def analyze(
        self,
        error_mask: torch.Tensor,
        label: torch.Tensor,
        prediction: torch.Tensor
    ) -> dict:
        """
        分析错误区域
        
        Args:
            error_mask: 错误掩码 (B, 1, H, W)
            label: 标签 (B, H, W)
            prediction: 预测概率 (B, C, H, W)
            
        Returns:
            dict: 分析结果
        """
        pred_class = prediction.argmax(dim=1)
        
        # 总像素数
        valid_mask = (label != self.ignore_index)
        total_pixels = valid_mask.sum().item()
        
        # 错误像素数
        error_pixels = error_mask.sum().item()
        
        # 错误率
        error_rate = error_pixels / total_pixels if total_pixels > 0 else 0
        
        # 类别级错误分析
        num_classes = prediction.shape[1]
        class_errors = {}
        
        for c in range(num_classes):
            class_mask = (label == c) & valid_mask
            class_error_mask = error_mask.squeeze(1) * class_mask
            class_total = class_mask.sum().item()
            class_error = class_error_mask.sum().item()
            
            if class_total > 0:
                class_errors[c] = {
                    'total': class_total,
                    'error': class_error,
                    'error_rate': class_error / class_total
                }
        
        return {
            'total_pixels': total_pixels,
            'error_pixels': error_pixels,
            'error_rate': error_rate,
            'class_errors': class_errors
        }
    
    def get_error_statistics(
        self,
        error_masks: List[torch.Tensor]
    ) -> dict:
        """
        计算多个样本的错误统计
        
        Args:
            error_masks: 错误掩码列表
            
        Returns:
            dict: 统计结果
        """
        error_rates = []
        for mask in error_masks:
            rate = mask.sum().item() / mask.numel()
            error_rates.append(rate)
        
        return {
            'mean_error_rate': np.mean(error_rates),
            'std_error_rate': np.std(error_rates),
            'max_error_rate': np.max(error_rates),
            'min_error_rate': np.min(error_rates)
        }
