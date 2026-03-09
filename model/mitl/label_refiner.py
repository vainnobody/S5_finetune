"""
标签优化主类

整合置信度评估、错误检测和标签融合，实现完整的标签优化流程。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional, Dict, List
from PIL import Image

from .confidence_eval import ConfidenceEvaluator
from .error_detector import ErrorDetector, ErrorRegionAnalyzer
from .label_fusion import LabelFusion, compute_fusion_quality


class LabelRefiner:
    """
    标签优化器
    
    整合MITL流水线的核心组件，实现端到端的标签优化。
    """
    
    def __init__(
        self,
        num_classes: int,
        confidence_method: str = 'mc_dropout',
        confidence_threshold: float = 0.8,
        fusion_method: str = 'selective',
        n_samples: int = 10,
        ignore_index: int = 255,
        device: str = 'cuda',
        crop_size: int = 512
    ):
        """
        Args:
            num_classes: 类别数
            confidence_method: 置信度评估方法
            confidence_threshold: 置信度阈值
            fusion_method: 融合方法
            n_samples: MC-Dropout采样次数
            ignore_index: 忽略的标签值
            device: 设备
            crop_size: 滑动窗口裁剪大小
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        self.crop_size = crop_size
        
        # 初始化组件
        self.confidence_eval = ConfidenceEvaluator(
            method=confidence_method,
            n_samples=n_samples,
            crop_size=crop_size
        )
        
        self.error_detector = ErrorDetector(
            confidence_threshold=confidence_threshold,
            ignore_index=ignore_index
        )
        
        self.label_fusion = LabelFusion(
            method=fusion_method,
            confidence_threshold=confidence_threshold,
            ignore_index=ignore_index
        )
        
        self.analyzer = ErrorRegionAnalyzer(ignore_index=ignore_index)
    
    def refine_single(
        self,
        model: nn.Module,
        image: torch.Tensor,
        label: torch.Tensor,
        return_details: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        优化单个样本的标签
        
        Args:
            model: 分割模型
            image: 输入图像 (B, C, H, W)
            label: 原始标签 (B, H, W)
            return_details: 是否返回详细信息
            
        Returns:
            refined_label: 优化后的标签 (B, H, W)
            details: 详细信息 (可选)
        """
        with torch.no_grad():
            # 1. 置信度评估
            prediction, confidence, uncertainty = self.confidence_eval.evaluate(
                model, image, return_all=True
            )
            
            # 2. 错误检测
            error_mask = self.error_detector.detect(
                prediction, label, confidence, uncertainty, method='comprehensive'
            )
            
            # 3. 标签融合
            refined_label = self.label_fusion.fuse(
                label, prediction, confidence, error_mask
            )
        
        if return_details:
            details = {
                'prediction': prediction,
                'confidence': confidence,
                'uncertainty': uncertainty,
                'error_mask': error_mask,
                'analysis': self.analyzer.analyze(error_mask, label, prediction)
            }
            return refined_label, details
        
        return refined_label, None
    
    def refine_batch(
        self,
        model: nn.Module,
        dataloader,
        save_dir: Optional[str] = None,
        save_confidence_maps: bool = False,
        verbose: bool = True
    ) -> Dict:
        """
        批量优化标签
        
        Args:
            model: 分割模型
            dataloader: 数据加载器
            save_dir: 保存目录
            save_confidence_maps: 是否保存置信度图
            verbose: 是否显示进度条
            
        Returns:
            stats: 统计信息
        """
        model.eval()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(os.path.join(save_dir, 'refined_labels'), exist_ok=True)
            if save_confidence_maps:
                os.makedirs(os.path.join(save_dir, 'confidence_maps'), exist_ok=True)
        
        all_stats = []
        total_changes = 0
        total_pixels = 0
        
        iterator = tqdm(dataloader, desc='Refining labels') if verbose else dataloader
        
        for batch_idx, batch in enumerate(iterator):
            if len(batch) >= 2:
                images, labels = batch[0], batch[1]
                paths = batch[2] if len(batch) > 2 else None
            else:
                continue
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 优化标签
            refined_labels, details = self.refine_single(
                model, images, labels, return_details=True
            )
            
            # 统计变化
            valid_mask = (labels != self.ignore_index)
            changes = (labels != refined_labels) & valid_mask
            batch_changes = changes.sum().item()
            batch_pixels = valid_mask.sum().item()
            
            total_changes += batch_changes
            total_pixels += batch_pixels
            
            # 计算质量指标
            quality = compute_fusion_quality(
                labels, refined_labels, details['prediction']
            )
            
            all_stats.append({
                'change_rate': batch_changes / batch_pixels if batch_pixels > 0 else 0,
                'mean_confidence': details['confidence'].mean().item(),
                'error_rate': details['analysis']['error_rate']
            })
            
            # 保存结果
            if save_dir:
                for i in range(len(refined_labels)):
                    # 确定保存文件名
                    if paths is not None:
                        # paths可能是 "image_path mask_path" 格式，提取文件名
                        path_info = paths[i].split(' ')[0]  # 取图像路径
                        filename = os.path.basename(path_info)
                    else:
                        # 无路径时使用序号ID
                        filename = f'refined_{batch_idx * dataloader.batch_size + i:06d}.png'
                    
                    # 保存优化后的标签
                    label_path = os.path.join(save_dir, 'refined_labels', filename)
                    refined_label_np = refined_labels[i].cpu().numpy()
                    Image.fromarray(refined_label_np.astype(np.uint8)).save(label_path)
                    
                    # 保存置信度图
                    if save_confidence_maps:
                        conf_path = os.path.join(save_dir, 'confidence_maps', filename)
                        conf_np = (details['confidence'][i].squeeze().cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(conf_np).save(conf_path)
        
        # 汇总统计
        overall_stats = {
            'total_samples': len(all_stats),
            'total_changes': total_changes,
            'total_pixels': total_pixels,
            'overall_change_rate': total_changes / total_pixels if total_pixels > 0 else 0,
            'mean_change_rate': np.mean([s['change_rate'] for s in all_stats]),
            'std_change_rate': np.std([s['change_rate'] for s in all_stats]),
            'mean_confidence': np.mean([s['mean_confidence'] for s in all_stats]),
            'mean_error_rate': np.mean([s['error_rate'] for s in all_stats])
        }
        
        # 保存统计信息
        if save_dir:
            with open(os.path.join(save_dir, 'refinement_stats.json'), 'w') as f:
                json.dump(overall_stats, f, indent=2)
        
        return overall_stats
    
    def iterative_refine(
        self,
        model: nn.Module,
        dataloader,
        num_iterations: int = 3,
        early_stop_threshold: float = 0.01,
        save_dir: Optional[str] = None
    ) -> Tuple[Dict, List[Dict]]:
        """
        迭代式标签优化
        
        Args:
            model: 分割模型
            dataloader: 数据加载器
            num_iterations: 最大迭代次数
            early_stop_threshold: 早停阈值
            save_dir: 保存目录
            
        Returns:
            final_stats: 最终统计信息
            history: 每次迭代的历史记录
        """
        history = []
        
        for iteration in range(num_iterations):
            iter_save_dir = None
            if save_dir:
                iter_save_dir = os.path.join(save_dir, f'iteration_{iteration + 1}')
            
            print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
            
            # 执行标签优化
            stats = self.refine_batch(
                model, dataloader, iter_save_dir, verbose=True
            )
            
            history.append(stats)
            
            # 检查是否早停
            if stats['overall_change_rate'] < early_stop_threshold:
                print(f"Early stopping: change rate {stats['overall_change_rate']:.4f} < {early_stop_threshold}")
                break
            
            # 更新数据加载器中的标签路径
            # 这里需要根据具体的数据集类进行实现
        
        final_stats = history[-1] if history else {}
        
        # 保存完整历史
        if save_dir:
            with open(os.path.join(save_dir, 'refinement_history.json'), 'w') as f:
                json.dump(history, f, indent=2)
        
        return final_stats, history


class IURNetRefiner(LabelRefiner):
    """
    IUR-Net风格的标签优化器
    
    实现Identify-Update-Refine三阶段标签优化流程。
    
    参考: IUR-Net: A Multi-Stage Framework for Label Refinement Tasks 
         in Noisy Remote Sensing Samples (Remote Sensing 2025)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_names = ['Identify', 'Update', 'Refine']
    
    def identify_stage(
        self,
        model: nn.Module,
        image: torch.Tensor,
        label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Identify阶段：识别错误区域
        
        Args:
            model: 分割模型
            image: 输入图像
            label: 原始标签
            
        Returns:
            prediction: 模型预测
            confidence: 置信度图
            error_mask: 错误区域掩码
        """
        # 使用综合方法评估置信度
        results = self.confidence_eval.evaluate(model, image, return_all=True)
        
        if isinstance(results, dict):
            prediction = results['prediction']
            confidence = results['max_prob_confidence']
        else:
            prediction, confidence, _ = results
        
        # 综合错误检测
        error_mask = self.error_detector.detect(
            prediction, label, confidence, method='comprehensive'
        )
        
        return prediction, confidence, error_mask
    
    def update_stage(
        self,
        label: torch.Tensor,
        prediction: torch.Tensor,
        error_mask: torch.Tensor,
        confidence: torch.Tensor
    ) -> torch.Tensor:
        """
        Update阶段：更新错误区域的标签
        
        Args:
            label: 原始标签
            prediction: 模型预测
            error_mask: 错误区域掩码
            confidence: 置信度图
            
        Returns:
            updated_label: 更新后的标签
        """
        # 只在错误区域且高置信度的地方更新
        pred_class = prediction.argmax(dim=1)
        
        # 高置信度的错误区域
        high_conf_error = (error_mask > 0.5) & (confidence.squeeze(1) > self.label_fusion.confidence_threshold)
        
        # 更新标签
        updated_label = label.clone()
        updated_label[high_conf_error.squeeze(1)] = pred_class[high_conf_error.squeeze(1)]
        
        return updated_label
    
    def refine_stage(
        self,
        label: torch.Tensor,
        prediction: torch.Tensor,
        confidence: torch.Tensor
    ) -> torch.Tensor:
        """
        Refine阶段：精细化边界区域
        
        Args:
            label: 更新后的标签
            prediction: 模型预测
            confidence: 置信度图
            
        Returns:
            refined_label: 精细化后的标签
        """
        # 使用边界感知融合
        self.label_fusion.method = 'boundary_aware'
        refined_label = self.label_fusion.fuse(label, prediction, confidence)
        
        return refined_label
    
    def refine_single(
        self,
        model: nn.Module,
        image: torch.Tensor,
        label: torch.Tensor,
        return_details: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        执行完整的IUR流程
        """
        details = {'stages': {}}
        
        # Stage 1: Identify
        prediction, confidence, error_mask = self.identify_stage(model, image, label)
        details['stages']['identify'] = {
            'error_rate': error_mask.mean().item()
        }
        
        # Stage 2: Update
        updated_label = self.update_stage(label, prediction, error_mask, confidence)
        details['stages']['update'] = {
            'update_rate': ((updated_label != label) & (label != self.ignore_index)).sum().item() / 
                          ((label != self.ignore_index)).sum().item()
        }
        
        # Stage 3: Refine
        refined_label = self.refine_stage(updated_label, prediction, confidence)
        details['stages']['refine'] = {
            'refine_rate': ((refined_label != updated_label) & (updated_label != self.ignore_index)).sum().item() /
                          ((updated_label != self.ignore_index)).sum().item()
        }
        
        if return_details:
            details['prediction'] = prediction
            details['confidence'] = confidence
            details['error_mask'] = error_mask
            return refined_label, details
        
        return refined_label, None


def create_refiner(
    config: dict,
    num_classes: int,
    device: str = 'cuda',
    crop_size: int = 512
) -> LabelRefiner:
    """
    工厂函数：创建标签优化器
    
    Args:
        config: 配置字典
        num_classes: 类别数
        device: 设备
        crop_size: 滑动窗口裁剪大小
        
    Returns:
        refiner: 标签优化器实例
    """
    refiner_type = config.get('type', 'standard')
    
    common_kwargs = {
        'num_classes': num_classes,
        'confidence_method': config.get('confidence_method', 'mc_dropout'),
        'confidence_threshold': config.get('confidence_threshold', 0.8),
        'fusion_method': config.get('fusion_method', 'selective'),
        'n_samples': config.get('mc_dropout_samples', 10),
        'ignore_index': config.get('ignore_index', 255),
        'device': device,
        'crop_size': crop_size
    }
    
    if refiner_type == 'iurnet':
        return IURNetRefiner(**common_kwargs)
    else:
        return LabelRefiner(**common_kwargs)
