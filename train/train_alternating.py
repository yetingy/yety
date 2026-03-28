"""
交替训练策略 - 轮流训练检测和改写任务
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.model_config import ModelConfig
from config.training_config import TrainingConfig, DataConfig
from models.multitask_model import MultiTaskModel
from .train_detection import train_detection
from .train_rewriting import train_rewriting
import logging


def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def train_alternating(
    model_manager: MultiTaskModel,
    detection_data: Dict[str, str],
    rewriting_data: Dict[str, str],
    training_config: TrainingConfig,
    model_config: ModelConfig,
    iterations: int = None,
    callbacks: List[callable] = None
) -> Dict[str, Any]:
    """
    交替训练检测和改写任务
    
    策略：
    Iteration 1:
      1. 训练 detection (epochs_per_iter 轮)
      2. 训练 rewriting (epochs_per_iter 轮)
    Iteration 2:
      ...
    
    Args:
        model_manager: 多任务模型管理器
        detection_data: 包含 train/eval 路径的字典
        rewriting_data: 包含 train/eval 路径的字典
        training_config: 训练配置
        model_config: 模型配置
        iterations: 迭代次数（覆盖配置中的值）
        callbacks: 回调函数列表，每个迭代后调用
        
    Returns:
        训练历史记录
    """
    logger = setup_logging()
    
    if iterations is None:
        iterations = training_config.alternating_iterations
    
    logger.info(f"Starting alternating training: {iterations} iterations")
    logger.info(f"Detection epochs per iter: {training_config.detection_epochs_per_iter}")
    logger.info(f"Rewriting epochs per iter: {training_config.rewriting_epochs_per_iter}")
    
    history = {
        "detection_metrics": [],
        "rewriting_metrics": [],
        "checkpoints": []
    }
    
    for iteration in range(iterations):
        print("\n" + "="*60)
        print(f"Iteration {iteration + 1}/{iterations}")
        print("="*60)
        
        # ============ 训练检测任务 ============
        print("\n[1/2] Training DETECTION...")
        detection_metrics = train_detection(
            model_manager=model_manager,
            train_data_path=detection_data["train"],
            eval_data_path=detection_data["eval"],
            training_config=training_config,
            model_config=model_config,
            output_dir=f"{training_config.output_dir}/detection_iter{iteration}"
        )
        
        history["detection_metrics"].append(detection_metrics)
        print(f"  [OK] Detection metrics: {detection_metrics}")
        
        # ============ 训练改写任务 ============
        print("\n[2/2] Training REWRITING...")
        rewriting_metrics = train_rewriting(
            model_manager=model_manager,
            train_data_path=rewriting_data["train"],
            eval_data_path=rewriting_data["eval"],
            training_config=training_config,
            model_config=model_config,
            output_dir=f"{training_config.output_dir}/rewriting_iter{iteration}"
        )
        
        history["rewriting_metrics"].append(rewriting_metrics)
        print(f"  [OK] Rewriting metrics: {rewriting_metrics}")
        
        # 保存迭代检查点
        checkpoint_path = f"{training_config.output_dir}/joint_iter{iteration}"
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        
        model_manager.save_adapter("detection", f"{checkpoint_path}/detection")
        model_manager.save_adapter("rewriting", f"{checkpoint_path}/rewriting")
        
        history["checkpoints"].append({
            "iteration": iteration,
            "detection_path": f"{checkpoint_path}/detection",
            "rewriting_path": f"{checkpoint_path}/rewriting",
            "detection_metrics": detection_metrics,
            "rewriting_metrics": rewriting_metrics
        })
        
        print(f"  [DATA] Checkpoint saved to {checkpoint_path}")
        
        # 执行回调（如评估、日志记录等）
        if callbacks:
            for callback in callbacks:
                callback(iteration, history)
    
    print("\n" + "="*60)
    print("[OK] Alternating training COMPLETED!")
    print("="*60)
    
    return history


def train_alternating_with_validation(
    model_manager: MultiTaskModel,
    detection_data: Dict[str, str],
    rewriting_data: Dict[str, str],
    training_config: TrainingConfig,
    model_config: ModelConfig,
    **kwargs
) -> Dict[str, Any]:
    """
    带验证的交替训练（每轮后评估两个任务）
    """
    logger = logging.getLogger(__name__)
    logger.info("Running alternating training with validation after each epoch...")
    
    # 这里可以添加更复杂的验证逻辑
    # 比如每轮后分别评估两个任务的性能，如果某个任务性能下降太多则调整学习率等
    
    return train_alternating(
        model_manager=model_manager,
        detection_data=detection_data,
        rewriting_data=rewriting_data,
        training_config=training_config,
        model_config=model_config,
        **kwargs
    )


if __name__ == "__main__":
    from config.model_config import ModelConfig
    from data.preprocess import prepare_dummy_datasets
    
    # 准备示例数据
    print("Preparing dummy datasets...")
    prepare_dummy_datasets()
    
    # 加载配置
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # 初始化模型管理器
    print("\nInitializing MultiTaskModel...")
    model_manager = MultiTaskModel(
        base_model_name=model_config.base_model_name,
        model_config=model_config,
        device_map=model_config.device_map
    )
    
    # 数据路径
    detection_data = {
        "train": "data/detection/train.json",
        "eval": "data/detection/eval.json"
    }
    rewriting_data = {
        "train": "data/rewriting/train.json",
        "eval": "data/rewriting/eval.json"
    }
    
    # 开始交替训练
    history = train_alternating(
        model_manager=model_manager,
        detection_data=detection_data,
        rewriting_data=rewriting_data,
        training_config=training_config,
        model_config=model_config,
        iterations=2  # 测试用2轮
    )
    
    print("\nTraining history:")
    print(f"Detection checkpoints: {len(history['detection_metrics'])}")
    print(f"Rewriting checkpoints: {len(history['rewriting_metrics'])}")
    print(f"Total joint checkpoints: {len(history['checkpoints'])}")
