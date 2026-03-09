"""
MITL (Model-in-the-Loop) 标签优化模块

该模块实现了基于模型内循环的标签优化流水线，用于遥感图像语义分割数据集的标签清洗和优化。

主要组件:
- ConfidenceEvaluator: 置信度评估模块
- ErrorDetector: 错误区域检测模块
- LabelFusion: 标签融合模块
- LabelRefiner: 标签优化主类
- IURNetRefiner: IUR-Net风格的标签优化器
- create_refiner: 工厂函数
"""

from .confidence_eval import ConfidenceEvaluator, TemperatureScaler
from .error_detector import ErrorDetector, MultiScaleErrorAwareLocalization, ErrorRegionAnalyzer
from .label_fusion import LabelFusion, SoftLabelGenerator, IterativeLabelRefiner
from .label_refiner import LabelRefiner, IURNetRefiner, create_refiner

__all__ = [
    'ConfidenceEvaluator',
    'TemperatureScaler',
    'ErrorDetector',
    'MultiScaleErrorAwareLocalization',
    'ErrorRegionAnalyzer',
    'LabelFusion',
    'SoftLabelGenerator',
    'IterativeLabelRefiner',
    'LabelRefiner',
    'IURNetRefiner',
    'create_refiner'
]

__version__ = '1.0.0'
