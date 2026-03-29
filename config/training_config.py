"""
训练超参数配置
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """训练超参数"""
    
    # 通用训练参数
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1  # Reduced for 4GB GPU
    per_device_eval_batch_size: int = 2   # Reduced for 4GB GPU
    gradient_accumulation_steps: int = 16  # Increased to maintain effective batch size
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    eval_strategy: str = "steps"
    save_total_limit: int = 2
    report_to: str = "none"
    push_to_hub: bool = False

    # 检测任务特定参数
    detection_max_length: int = 256  # Reduced for 4GB GPU

    # 改写任务特定参数
    rewriting_max_length: int = 256  # Reduced for 4GB GPU
    rewriting_max_new_tokens: int = 256
    rewriting_temperature: float = 0.7
    rewriting_do_sample: bool = True
    
    # 交替训练参数
    alternating_iterations: int = 5
    detection_epochs_per_iter: int = 1
    rewriting_epochs_per_iter: int = 1


@dataclass
class DataConfig:
    """数据配置"""
    
    # 检测任务数据
    detection_train_path: str = "data/detection/train.json"
    detection_eval_path: str = "data/detection/eval.json"
    
    # 改写任务数据
    rewriting_train_path: str = "data/rewriting/train.json"
    rewriting_eval_path: str = "data/rewriting/eval.json"
    
    # 数据预处理
    max_samples_per_task: int = 10000  # 每个任务的最大样本数
    train_eval_split: float = 0.9  # 训练/验证集分割比例
