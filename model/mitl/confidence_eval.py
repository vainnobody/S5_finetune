"""
置信度评估模块

实现多种置信度评估方法:
1. Monte Carlo Dropout - 通过多次随机推理评估不确定性
2. 预测熵 - 基于信息熵的不确定性度量
3. Temperature Scaling - 校准预测概率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def sliding_window_forward(
    model: nn.Module,
    image: torch.Tensor,
    crop_size: int = 512,
    step: int = 512
) -> torch.Tensor:
    """
    滑动窗口前向传播
    
    Args:
        model: 分割模型
        image: 输入图像 [B, C, H, W]
        crop_size: 裁剪窗口大小
        step: 滑动步长
    
    Returns:
        logits: 预测logits [B, C, H, W]
    """
    b, _, h, w = image.shape
    
    # 小图直接推理
    if h <= crop_size and w <= crop_size:
        return model(image)
    
    # 大图使用滑动窗口
    final = torch.zeros(b, model.num_classes if hasattr(model, 'num_classes') else 22, h, w).to(image.device)
    count = torch.zeros(b, 1, h, w).to(image.device)
    
    row = 0
    while row * step < h:
        col = 0
        while col * step < w:
            h_start = min(row * step, h - crop_size)
            w_start = min(col * step, w - crop_size)
            h_end = min(h_start + crop_size, h)
            w_end = min(w_start + crop_size, w)
            
            sub_input = image[:, :, h_start:h_end, w_start:w_end]
            
            # 处理padding
            if sub_input.shape[-2] < crop_size or sub_input.shape[-1] < crop_size:
                sub_input = F.pad(sub_input, 
                                  (0, crop_size - sub_input.shape[-1],
                                   0, crop_size - sub_input.shape[-2]),
                                  mode='constant', value=0)
            
            sub_pred = model(sub_input)
            if isinstance(sub_pred, dict):
                sub_pred = sub_pred['out']
            
            # 去除padding
            sub_pred = sub_pred[:, :, :h_end-h_start, :w_end-w_start]
            
            final[:, :, h_start:h_end, w_start:w_end] += sub_pred
            count[:, :, h_start:h_end, w_start:w_end] += 1
            
            col += 1
        row += 1
    
    return final / count.clamp(min=1)


class ConfidenceEvaluator:
    """
    置信度评估器
    
    支持多种置信度评估方法，用于识别模型预测的可靠程度。
    """
    
    def __init__(
        self,
        method: str = 'mc_dropout',
        n_samples: int = 10,
        temperature: float = 1.0,
        dropout_rate: float = 0.1,
        crop_size: int = 512
    ):
        """
        Args:
            method: 置信度评估方法 ('mc_dropout', 'entropy', 'max_prob', 'all')
            n_samples: MC-Dropout采样次数
            temperature: Temperature Scaling参数
            dropout_rate: Dropout率
            crop_size: 滑动窗口裁剪大小
        """
        self.method = method
        self.n_samples = n_samples
        self.temperature = temperature
        self.dropout_rate = dropout_rate
        self.crop_size = crop_size
    
    def evaluate(
        self,
        model: nn.Module,
        image: torch.Tensor,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        评估预测置信度
        
        Args:
            model: 分割模型
            image: 输入图像 (B, C, H, W)
            return_all: 是否返回所有评估结果
            
        Returns:
            prediction: 模型预测 (B, C, H, W)
            confidence: 置信度图 (B, 1, H, W)
            uncertainty: 不确定性图 (B, 1, H, W) (可选)
        """
        if self.method == 'mc_dropout':
            return self.mc_dropout_inference(model, image, return_all)
        elif self.method == 'entropy':
            return self.entropy_based_inference(model, image, return_all)
        elif self.method == 'max_prob':
            return self.max_prob_inference(model, image, return_all)
        elif self.method == 'all':
            return self.comprehensive_inference(model, image)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def mc_dropout_inference(
        self,
        model: nn.Module,
        image: torch.Tensor,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Monte Carlo Dropout推理
        
        通过多次随机前向传播评估模型预测的不确定性。
        支持滑动窗口处理大图。
        
        Args:
            model: 分割模型
            image: 输入图像 (B, C, H, W)
            return_all: 是否返回额外信息
            
        Returns:
            mean_pred: 平均预测概率 (B, C, H, W)
            confidence: 置信度图 (B, 1, H, W)
            uncertainty: 不确定性图 (B, 1, H, W)
        """
        model.train()  # 启用dropout
        
        b, _, h, w = image.shape
        use_sliding_window = h > self.crop_size or w > self.crop_size
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                if use_sliding_window:
                    logits = sliding_window_forward(model, image, self.crop_size, self.crop_size)
                else:
                    logits = model(image)
                if isinstance(logits, dict):
                    logits = logits['out']
                prob = F.softmax(logits / self.temperature, dim=1)
                predictions.append(prob)
        
        # 计算统计量
        predictions_stack = torch.stack(predictions, dim=0)  # (n_samples, B, C, H, W)
        mean_pred = predictions_stack.mean(dim=0)  # (B, C, H, W)
        variance = predictions_stack.var(dim=0)    # (B, C, H, W)
        
        # 置信度 = 最大类别概率的均值
        confidence = mean_pred.max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
        
        # 不确定性 = 预测方差的总和（跨类别）
        uncertainty = variance.sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        if return_all:
            return mean_pred, confidence, uncertainty
        return mean_pred, confidence, None
    
    def entropy_based_inference(
        self,
        model: nn.Module,
        image: torch.Tensor,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        基于预测熵的置信度评估
        
        使用信息熵度量预测的不确定性。
        支持滑动窗口处理大图。
        
        Args:
            model: 分割模型
            image: 输入图像 (B, C, H, W)
            return_all: 是否返回额外信息
            
        Returns:
            prediction: 预测概率 (B, C, H, W)
            confidence: 置信度图 (B, 1, H, W)
            entropy: 熵图 (B, 1, H, W)
        """
        model.eval()
        
        b, _, h, w = image.shape
        use_sliding_window = h > self.crop_size or w > self.crop_size
        
        with torch.no_grad():
            if use_sliding_window:
                logits = sliding_window_forward(model, image, self.crop_size, self.crop_size)
            else:
                logits = model(image)
            if isinstance(logits, dict):
                logits = logits['out']
            prediction = F.softmax(logits / self.temperature, dim=1)
        
        # 计算熵
        entropy = self.compute_entropy(prediction)  # (B, 1, H, W)
        
        # 置信度 = 1 - 归一化熵
        max_entropy = np.log(prediction.shape[1])  # 最大可能熵
        confidence = 1 - (entropy / max_entropy)
        
        if return_all:
            return prediction, confidence, entropy
        return prediction, confidence, None
    
    def max_prob_inference(
        self,
        model: nn.Module,
        image: torch.Tensor,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        基于最大概率的置信度评估
        
        最简单的方法：直接使用最大类别概率作为置信度。
        支持滑动窗口处理大图。
        
        Args:
            model: 分割模型
            image: 输入图像 (B, C, H, W)
            return_all: 是否返回额外信息
            
        Returns:
            prediction: 预测概率 (B, C, H, W)
            confidence: 置信度图 (B, 1, H, W)
            None
        """
        model.eval()
        
        b, _, h, w = image.shape
        use_sliding_window = h > self.crop_size or w > self.crop_size
        
        with torch.no_grad():
            if use_sliding_window:
                logits = sliding_window_forward(model, image, self.crop_size, self.crop_size)
            else:
                logits = model(image)
            if isinstance(logits, dict):
                logits = logits['out']
            prediction = F.softmax(logits / self.temperature, dim=1)
        
        # 置信度 = 最大类别概率
        confidence = prediction.max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
        
        return prediction, confidence, None
    
    def comprehensive_inference(
        self,
        model: nn.Module,
        image: torch.Tensor
    ) -> dict:
        """
        综合置信度评估
        
        同时计算多种置信度指标。
        
        Args:
            model: 分割模型
            image: 输入图像 (B, C, H, W)
            
        Returns:
            dict: 包含所有置信度指标
        """
        model.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = model(image)
                if isinstance(logits, dict):
                    logits = logits['out']
                prob = F.softmax(logits / self.temperature, dim=1)
                predictions.append(prob)
        
        predictions_stack = torch.stack(predictions, dim=0)
        mean_pred = predictions_stack.mean(dim=0)
        variance = predictions_stack.var(dim=0)
        
        # 最大概率置信度
        max_prob_conf = mean_pred.max(dim=1, keepdim=True)[0]
        
        # 熵
        entropy = self.compute_entropy(mean_pred)
        
        # 方差不确定性
        variance_unc = variance.sum(dim=1, keepdim=True)
        
        # 互信息 (Mutual Information)
        mutual_info = self.compute_mutual_information(predictions_stack)
        
        return {
            'prediction': mean_pred,
            'max_prob_confidence': max_prob_conf,
            'entropy': entropy,
            'variance_uncertainty': variance_unc,
            'mutual_information': mutual_info,
            'samples': predictions_stack if self.n_samples <= 20 else None
        }
    
    @staticmethod
    def compute_entropy(prob_map: torch.Tensor) -> torch.Tensor:
        """
        计算预测熵
        
        Args:
            prob_map: 预测概率图 (B, C, H, W)
            
        Returns:
            entropy: 熵图 (B, 1, H, W)
        """
        entropy = -torch.sum(prob_map * torch.log(prob_map + 1e-8), dim=1, keepdim=True)
        return entropy
    
    @staticmethod
    def compute_mutual_information(predictions_stack: torch.Tensor) -> torch.Tensor:
        """
        计算互信息
        
        MI = H[E[p(y|x)]] - E[H[p(y|x)]]
        
        Args:
            predictions_stack: 多次预测的堆叠 (n_samples, B, C, H, W)
            
        Returns:
            mutual_info: 互信息图 (B, 1, H, W)
        """
        # 平均预测的熵
        mean_pred = predictions_stack.mean(dim=0)
        entropy_mean = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=1, keepdim=True)
        
        # 每个样本熵的平均
        individual_entropies = -torch.sum(
            predictions_stack * torch.log(predictions_stack + 1e-8), 
            dim=2, keepdim=True
        )
        mean_entropy = individual_entropies.mean(dim=0)
        
        mutual_info = entropy_mean - mean_entropy
        return mutual_info


class TemperatureScaler(nn.Module):
    """
    Temperature Scaling 校准器
    
    用于校准模型的预测概率，使其更加准确反映真实置信度。
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        应用temperature scaling
        
        Args:
            logits: 模型输出logits (B, C, H, W)
            
        Returns:
            校准后的logits
        """
        return logits / self.temperature
    
    def calibrate(self, model: nn.Module, val_loader, device: str = 'cuda'):
        """
        在验证集上校准temperature参数
        
        Args:
            model: 分割模型
            val_loader: 验证数据加载器
            device: 设备
        """
        model.eval()
        self.to(device)
        
        # 收集验证集上的logits和标签
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for image, label in val_loader:
                image = image.to(device)
                label = label.to(device)
                
                logits = model(image)
                if isinstance(logits, dict):
                    logits = logits['out']
                
                all_logits.append(logits)
                all_labels.append(label)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 优化temperature参数
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(
                self.forward(all_logits), 
                all_labels,
                ignore_index=255
            )
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        return self.temperature.item()


def compute_class_confidence(
    prediction: torch.Tensor,
    class_idx: int
) -> torch.Tensor:
    """
    计算特定类别的置信度
    
    Args:
        prediction: 预测概率 (B, C, H, W)
        class_idx: 类别索引
        
    Returns:
        class_confidence: 该类别的置信度图 (B, 1, H, W)
    """
    return prediction[:, class_idx:class_idx+1, :, :]


def compute_pixel_confidence_stats(
    confidence_map: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> dict:
    """
    计算置信度统计信息
    
    Args:
        confidence_map: 置信度图 (B, 1, H, W)
        mask: 有效区域掩码 (可选)
        
    Returns:
        dict: 统计信息
    """
    if mask is not None:
        confidence_values = confidence_map[mask]
    else:
        confidence_values = confidence_map.flatten()
    
    return {
        'mean': confidence_values.mean().item(),
        'std': confidence_values.std().item(),
        'min': confidence_values.min().item(),
        'max': confidence_values.max().item(),
        'median': confidence_values.median().item()
    }
