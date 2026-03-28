"""
检测任务训练脚本
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.model_config import ModelConfig
from config.training_config import TrainingConfig, DataConfig
from data.dataset import DetectionDataset, load_json_data
from models.multitask_model import MultiTaskModel
from transformers import AutoTokenizer, TrainingArguments
from .trainer import DetectionTrainer, compute_metrics_detection
import logging


def setup_logging(log_file: str = None):
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_detection(
    model_manager: MultiTaskModel,
    train_data_path: str,
    eval_data_path: str,
    training_config: TrainingConfig,
    model_config: ModelConfig,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    训练检测任务
    
    Args:
        model_manager: 多任务模型管理器
        train_data_path: 训练数据路径
        eval_data_path: 评估数据路径
        training_config: 训练配置
        model_config: 模型配置
        output_dir: 输出目录（可选）
        
    Returns:
        训练结果
    """
    logger = setup_logging()
    
    # 设置输出目录
    if output_dir is None:
        output_dir = training_config.output_dir + "/detection"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting detection training, output: {output_dir}")
    
    # 1. 加载 tokenizer
    logger.info("Loading tokenizer...")
    model_path = model_config.get_model_path()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 加载数据
    logger.info(f"Loading data from {train_data_path} and {eval_data_path}")
    train_data = load_json_data(train_data_path)
    eval_data = load_json_data(eval_data_path)
    
    logger.info(f"Data sizes: train={len(train_data)}, eval={len(eval_data)}")
    
    # 3. 创建 dataset
    train_dataset = DetectionDataset(
        train_data,
        tokenizer,
        max_length=training_config.detection_max_length
    )
    eval_dataset = DetectionDataset(
        eval_data,
        tokenizer,
        max_length=training_config.detection_max_length
    )
    
    # 4. 获取或创建检测 adapter
    if "detection" not in model_manager.list_adapters():
        logger.info("Creating detection adapter...")
        model_manager.add_adapter(
            "detection",
            model_config.detection_lora,
            task_type="cls"
        )
    
    # 5. 设置活跃 adapter
    model = model_manager.set_active_adapter("detection")
    model.train()
    
    # 6. 配置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        lr_scheduler_type=training_config.lr_scheduler_type,
        warmup_ratio=training_config.warmup_ratio,
        weight_decay=training_config.weight_decay,
        fp16=training_config.fp16,
        logging_dir=f"{output_dir}/logs",
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        eval_steps=training_config.eval_steps,
        eval_strategy=training_config.eval_strategy,
        save_total_limit=training_config.save_total_limit,
        report_to=training_config.report_to,
        push_to_hub=training_config.push_to_hub,
        remove_unused_columns=False,  # 重要：不要删除未使用的列（如 labels）
    )
    
    # 7. 创建 Trainer
    trainer = DetectionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_detection,
    )
    
    # 8. 开始训练
    logger.info("Starting training loop...")
    train_result = trainer.train()
    
    # 9. 保存模型
    logger.info("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 10. 保存 adapter 到 model_manager
    model_manager.save_adapter("detection", output_dir)
    
    logger.info("[OK] Detection training completed!")
    return train_result.metrics


if __name__ == "__main__":
    # 示例用法
    from config.model_config import ModelConfig
    
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # 初始化模型管理器
    model_manager = MultiTaskModel(
        base_model_name=model_config.base_model_name,
        model_config=model_config,
        device_map=model_config.device_map
    )
    
    # 训练检测任务
    metrics = train_detection(
        model_manager=model_manager,
        train_data_path="data/detection/train.json",
        eval_data_path="data/detection/eval.json",
        training_config=training_config,
        model_config=model_config,
        output_dir="checkpoints/detection"
    )
    
    print("Training metrics:", metrics)
