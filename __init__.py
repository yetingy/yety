"""
Yety - AIGC Detection and Rewriting with Dual LoRA
"""

__version__ = "0.1.0"
__author__ = "Yi Xiaoting"
__description__ = "Academic paper AIGC detection and rewriting tool using dual LoRA"

from .models.multitask_model import MultiTaskModel
from .infer.predictor import YetyPredictor

__all__ = [
    "MultiTaskModel",
    "YetyPredictor"
]
