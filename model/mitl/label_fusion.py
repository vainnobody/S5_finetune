"""
标签融合模块

实现多种标签融合策略:
1. 选择性替换 - 高置信度区域用预测替换
2. 加权融合 - 软标签融合
3. 边界感知融合 - 保持边界完整性
4. 类别平衡融合 - 处理类别不平衡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union


class LabelFusion:
    """
    标签融合器
    
    将原始标签与模型预测融合，生成优化后的标签。
    """
    
    def __init__(
        self,
        method: str = 'selective',
        confidence_threshold: float = 0.9,
        alpha: float = 0.5,
        ignore_index: int = 255,
        smooth_boundary: bool = True
    ):
        """
        Args:
            method: 融合方法 ('selective', 'weighted', 'boundary_aware', 'class_balanced')
            confidence_threshold: 选择性替换的置信度阈值
            alpha: 加权融合的权重 (原标签权重)
            ignore_index: 忽略的标签值
            smooth_boundary: 是否平滑边界
        """
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.smooth_boundary = smooth_boundary
    
    def fuse(
        self,
        original_label: torch.Tensor,
        prediction: torch.Tensor,
        confidence: torch.Tensor,
        error_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        融合原始标签与预测
        
        Args:
            original_label: 原始标签 (B, H, W)
            prediction: 预测概率 (B, C, H, W)
            confidence: 置信度图 (B, 1, H, W)
            error_mask: 错误区域掩码 (B, 1, H, W) (可选)
            
        Returns:
            new_label: 融合后的标签 (B, H, W)
        """
        if self.method == 'selective':
            return self.selective_replacement(
                original_label, prediction, confidence, error_mask
            )
        elif self.method == 'weighted':
            return self.weighted_fusion(original_label, prediction)
        elif self.method == 'boundary_aware':
            return self.boundary_aware_fusion(
                original_label, prediction, confidence
            )
        elif self.method == 'class_balanced':
            return self.class_balanced_fusion(
                original_label, prediction, confidence
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def selective_replacement(
        self,
        original_label: torch.Tensor,
        prediction: torch.Tensor,
        confidence: torch.Tensor,
        error_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        选择性替换策略
        
        在高置信度区域用模型预测替换原始标签。
        
        Args:
            original_label: 原始标签 (B, H, W)
            prediction: 预测概率 (B, C, H, W)
            confidence: 置信度图 (B, 1, H, W)
            error_mask: 错误区域掩码 (B, 1, H, W)
            
        Returns:
            new_label: 新标签 (B, H, W)
        """
        # 获取预测类别
        pred_class = prediction.argmax(dim=1)  # (B, H, W)
        
        # 有效区域掩码
        valid_mask = (original_label != self.ignore_index)
        
        # 高置信度区域
        high_conf_mask = (confidence.squeeze(1) > self.confidence_threshold) & valid_mask
        
        # 如果有错误掩码，只在错误区域替换
        if error_mask is not None:
            replace_mask = high_conf_mask & (error_mask.squeeze(1) > 0.5)
        else:
            replace_mask = high_conf_mask
        
        # 创建新标签
        new_label = original_label.clone()
        new_label[replace_mask] = pred_class[replace_mask]
        
        return new_label
    
    def weighted_fusion(
        self,
        original_label: torch.Tensor,
        prediction: torch.Tensor
    ) -> torch.Tensor:
        """
        加权融合策略
        
        将原始标签的one-hot编码与预测概率加权融合。
        
        Args:
            original_label: 原始标签 (B, H, W)
            prediction: 预测概率 (B, C, H, W)
            
        Returns:
            new_label: 新标签 (B, H, W)
        """
        num_classes = prediction.shape[1]
        
        # 原标签one-hot编码
        valid_mask = (original_label != self.ignore_index)
        original_onehot = F.one_hot(
            original_label.clamp(0, num_classes - 1), 
            num_classes=num_classes
        ).permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # 加权融合
        fused = self.alpha * original_onehot + (1 - self.alpha) * prediction
        
        # 取最大值作为最终标签
        new_label = fused.argmax(dim=1)
        
        # 保持忽略区域
        new_label[~valid_mask] = self.ignore_index
        
        return new_label
    
    def boundary_aware_fusion(
        self,
        original_label: torch.Tensor,
        prediction: torch.Tensor,
        confidence: torch.Tensor
    ) -> torch.Tensor:
        """
        边界感知融合策略
        
        在物体内部区域更容易接受预测，边界区域保持原标签。
        
        Args:
            original_label: 原始标签 (B, H, W)
            prediction: 预测概率 (B, C, H, W)
            confidence: 置信度图 (B, 1, H, W)
            
        Returns:
            new_label: 新标签 (B, H, W)
        """
        # 提取边界
        boundary_mask = self._extract_boundary(original_label)
        
        # 内部区域掩码
        interior_mask = ~boundary_mask
        
        # 预测类别
        pred_class = prediction.argmax(dim=1)
        
        # 有效区域
        valid_mask = (original_label != self.ignore_index)
        
        # 在内部区域的高置信度位置替换
        replace_mask = (
            interior_mask & 
            (confidence.squeeze(1) > self.confidence_threshold) & 
            valid_mask
        )
        
        # 创建新标签
        new_label = original_label.clone()
        new_label[replace_mask] = pred_class[replace_mask]
        
        # 边界区域平滑
        if self.smooth_boundary:
            new_label = self._smooth_boundary(new_label, prediction)
        
        return new_label
    
    def class_balanced_fusion(
        self,
        original_label: torch.Tensor,
        prediction: torch.Tensor,
        confidence: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        类别平衡融合策略
        
        对少数类使用更高的替换阈值，避免过度修正。
        
        Args:
            original_label: 原始标签 (B, H, W)
            prediction: 预测概率 (B, C, H, W)
            confidence: 置信度图 (B, 1, H, W)
            class_weights: 类别权重 (C,) (可选)
            
        Returns:
            new_label: 新标签 (B, H, W)
        """
        num_classes = prediction.shape[1]
        
        # 如果没有提供类别权重，计算标签分布
        if class_weights is None:
            class_weights = self._compute_class_weights(original_label, num_classes)
        
        # 预测类别
        pred_class = prediction.argmax(dim=1)
        
        # 有效区域
        valid_mask = (original_label != self.ignore_index)
        
        # 为每个类别计算自适应阈值
        new_label = original_label.clone()
        
        for c in range(num_classes):
            # 少数类使用更高阈值
            weight = class_weights[c]
            adaptive_threshold = self.confidence_threshold + (1 - weight) * 0.05
            
            # 该类别的替换区域
            class_mask = (original_label == c) & valid_mask
            high_conf = (confidence.squeeze(1) > adaptive_threshold) & class_mask
            
            new_label[high_conf] = pred_class[high_conf]
        
        return new_label
    
    @staticmethod
    def _extract_boundary(
        label: torch.Tensor,
        width: int = 2
    ) -> torch.Tensor:
        """
        提取标签边界
        
        Args:
            label: 标签 (B, H, W)
            width: 边界宽度
            
        Returns:
            boundary_mask: 边界掩码 (B, H, W)
        """
        label_float = label.unsqueeze(1).float()  # (B, 1, H, W)
        
        # 膨胀
        kernel_size = 2 * width + 1
        dilated = F.max_pool2d(
            label_float, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=width
        )
        
        # 腐蚀
        eroded = -F.max_pool2d(
            -label_float, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=width
        )
        
        # 边界 = 膨胀 != 腐蚀
        boundary = (dilated != eroded).squeeze(1)
        
        return boundary
    
    def _smooth_boundary(
        self,
        label: torch.Tensor,
        prediction: torch.Tensor
    ) -> torch.Tensor:
        """
        平滑标签边界
        
        Args:
            label: 标签 (B, H, W)
            prediction: 预测概率 (B, C, H, W)
            
        Returns:
            smoothed_label: 平滑后的标签 (B, H, W)
        """
        # 使用预测概率的加权平均来平滑边界
        num_classes = prediction.shape[1]
        
        # 边界区域
        boundary = self._extract_boundary(label)
        
        # 在边界区域使用软标签
        boundary_probs = prediction[:, :, boundary].mean(dim=2)  # (B, C)
        boundary_class = boundary_probs.argmax(dim=1)
        
        smoothed_label = label.clone()
        
        # 只在边界区域应用平滑
        for b in range(label.shape[0]):
            smoothed_label[b, boundary[b]] = boundary_class[b]
        
        return smoothed_label
    
    def _compute_class_weights(
        self,
        label: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """
        计算类别权重（逆频率）
        
        Args:
            label: 标签 (B, H, W)
            num_classes: 类别数
            
        Returns:
            weights: 类别权重 (C,)
        """
        valid_mask = (label != self.ignore_index)
        valid_label = label[valid_mask]
        
        # 计算每个类别的像素数
        class_counts = torch.zeros(num_classes, device=label.device)
        for c in range(num_classes):
            class_counts[c] = (valid_label == c).sum().float()
        
        # 计算权重（逆频率，归一化）
        total = class_counts.sum()
        weights = total / (num_classes * class_counts + 1e-8)
        weights = weights / weights.sum()  # 归一化
        
        return weights


class SoftLabelGenerator:
    """
    软标签生成器
    
    将硬标签转换为软标签，用于更平滑的优化。
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        ignore_index: int = 255
    ):
        """
        Args:
            smoothing: 标签平滑系数
            ignore_index: 忽略的标签值
        """
        self.smoothing = smoothing
        self.ignore_index = ignore_index
    
    def generate(
        self,
        label: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """
        生成软标签
        
        Args:
            label: 硬标签 (B, H, W)
            num_classes: 类别数
            
        Returns:
            soft_label: 软标签 (B, C, H, W)
        """
        valid_mask = (label != self.ignore_index)
        
        # One-hot编码
        soft_label = F.one_hot(
            label.clamp(0, num_classes - 1), 
            num_classes=num_classes
        ).permute(0, 3, 1, 2).float()
        
        # 应用标签平滑
        if self.smoothing > 0:
            soft_label = soft_label * (1 - self.smoothing) + \
                        self.smoothing / num_classes
        
        # 保持忽略区域
        soft_label = soft_label * valid_mask.unsqueeze(1).float()
        
        return soft_label
    
    def fuse_with_prediction(
        self,
        label: torch.Tensor,
        prediction: torch.Tensor,
        fusion_weight: float = 0.5
    ) -> torch.Tensor:
        """
        将软标签与预测融合
        
        Args:
            label: 硬标签 (B, H, W)
            prediction: 预测概率 (B, C, H, W)
            fusion_weight: 标签权重
            
        Returns:
            fused_soft_label: 融合后的软标签 (B, C, H, W)
        """
        num_classes = prediction.shape[1]
        soft_label = self.generate(label, num_classes)
        
        # 加权融合
        fused = fusion_weight * soft_label + (1 - fusion_weight) * prediction
        
        # 归一化
        fused = fused / (fused.sum(dim=1, keepdim=True) + 1e-8)
        
        return fused


class IterativeLabelRefiner:
    """
    迭代标签优化器
    
    在多次迭代中逐步优化标签。
    """
    
    def __init__(
        self,
        fusion: LabelFusion,
        num_iterations: int = 3,
        early_stop_threshold: float = 0.01
    ):
        """
        Args:
            fusion: 标签融合器
            num_iterations: 最大迭代次数
            early_stop_threshold: 早停阈值（标签变化率）
        """
        self.fusion = fusion
        self.num_iterations = num_iterations
        self.early_stop_threshold = early_stop_threshold
    
    def refine(
        self,
        original_label: torch.Tensor,
        predictions: list,
        confidences: list
    ) -> Tuple[torch.Tensor, dict]:
        """
        迭代优化标签
        
        Args:
            original_label: 原始标签 (B, H, W)
            predictions: 每次迭代的预测列表
            confidences: 每次迭代的置信度列表
            
        Returns:
            refined_label: 优化后的标签 (B, H, W)
            history: 优化历史记录
        """
        current_label = original_label.clone()
        history = {'changes': [], 'iterations': 0}
        
        for i in range(self.num_iterations):
            pred = predictions[i] if i < len(predictions) else predictions[-1]
            conf = confidences[i] if i < len(confidences) else confidences[-1]
            
            # 融合标签
            new_label = self.fusion.fuse(current_label, pred, conf)
            
            # 计算变化率
            valid_mask = (current_label != self.fusion.ignore_index)
            changes = (new_label != current_label) & valid_mask
            change_rate = changes.sum().float() / valid_mask.sum().float()
            
            history['changes'].append(change_rate.item())
            
            # 更新标签
            current_label = new_label
            history['iterations'] = i + 1
            
            # 早停检查
            if change_rate < self.early_stop_threshold:
                break
        
        return current_label, history


def compute_fusion_quality(
    original_label: torch.Tensor,
    fused_label: torch.Tensor,
    prediction: torch.Tensor,
    ground_truth: Optional[torch.Tensor] = None
) -> dict:
    """
    计算融合质量指标
    
    Args:
        original_label: 原始标签 (B, H, W)
        fused_label: 融合后标签 (B, H, W)
        prediction: 预测概率 (B, C, H, W)
        ground_truth: 真实标签 (可选)
        
    Returns:
        dict: 质量指标
    """
    valid_mask = (original_label != 255)
    
    # 标签变化统计
    changes = (original_label != fused_label) & valid_mask
    change_rate = changes.sum().float() / valid_mask.sum().float()
    
    # 与预测的一致性
    pred_class = prediction.argmax(dim=1)
    original_consistency = ((original_label == pred_class) & valid_mask).sum().float() / valid_mask.sum().float()
    fused_consistency = ((fused_label == pred_class) & valid_mask).sum().float() / valid_mask.sum().float()
    
    metrics = {
        'change_rate': change_rate.item(),
        'original_prediction_consistency': original_consistency.item(),
        'fused_prediction_consistency': fused_consistency.item()
    }
    
    # 如果有真实标签
    if ground_truth is not None:
        original_correct = ((original_label == ground_truth) & valid_mask).sum().float() / valid_mask.sum().float()
        fused_correct = ((fused_label == ground_truth) & valid_mask).sum().float() / valid_mask.sum().float()
        
        metrics['original_accuracy'] = original_correct.item()
        metrics['fused_accuracy'] = fused_correct.item()
        metrics['improvement'] = (fused_correct - original_correct).item()
    
    return metrics
