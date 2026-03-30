"""
在 Cobal 上运行 - 只训练改写任务（使用 Unsloth）
在已训练好的检测模型基础上继续训练改写
"""

# 1. 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
import sys
from pathlib import Path

# 设置项目路径
PROJECT_PATH = "/content/drive/MyDrive/yety"
sys.path.insert(0, PROJECT_PATH)
os.chdir(PROJECT_PATH)

# 2. 导入 Unsloth 和其他模块
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TrainingArguments
from peft import LoraConfig, TaskType, PeftModel
import torch
import logging

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from data.dataset import RewritingDataset, load_json_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 3. 配置路径
BASE_MODEL_PATH = "Qwen/Qwen3.5-2B"  # 原始基础模型
DETECTION_CHECKPOINT = "/content/drive/MyDrive/yety/checkpoints/detection_iter0"
REWITING_OUTPUT = "/content/drive/MyDrive/yety/checkpoints/rewriting_iter0"

# 4. 训练改写任务
def train_rewriting_only():
    model_config = ModelConfig()
    training_config = TrainingConfig()

    # 1. 加载 tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 使用 Unsloth 加载原始基础模型
    logger.info(f"Loading base model: {BASE_MODEL_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_PATH,
        max_seq_length=training_config.rewriting_max_length,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    logger.info("[OK] Base model loaded (4bit quantized)")

    # 3. 加载之前训练好的检测 LoRA adapter
    logger.info(f"Loading detection LoRA from {DETECTION_CHECKPOINT}...")
    model = PeftModel.from_pretrained(
        model,
        DETECTION_CHECKPOINT,
        adapter_name="detection",
        is_trainable=False  # 检测 LoRA 不训练，只做参考
    )
    logger.info("[OK] Detection LoRA loaded (will be preserved during training)")

    # 4. 添加改写 LoRA adapter（新的，可训练）
    logger.info("Creating rewriting LoRA adapter...")

    rewriting_lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    model = model.add_adapter(rewriting_lora_config, adapter_name="rewriting")

    # 设置只有 rewritin adapter 可训练
    # 注意：PEFT 会自动处理，只有 rewritin 的参数会被更新
    model.print_trainable_parameters()
    logger.info("[OK] Rewriting LoRA adapter added")

    # 5. 启用梯度检查点以节省显存
    model.enable_gradient_checkpointing()

    # 6. 加载数据
    logger.info("Loading rewriting data...")
    train_data = load_json_data("data/rewriting/train.json")
    eval_data = load_json_data("data/rewriting/eval.json")
    logger.info(f"Data: train={len(train_data)}, eval={len(eval_data)}")

    train_dataset = RewritingDataset(
        train_data, tokenizer,
        max_length=training_config.rewriting_max_length,
        max_new_tokens=training_config.rewriting_max_new_tokens
    )
    eval_dataset = RewritingDataset(
        eval_data, tokenizer,
        max_length=training_config.rewriting_max_length,
        max_new_tokens=training_config.rewriting_max_new_tokens
    )

    # 7. 数据整理器
    from unsloth.trainers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=training_config.rewriting_max_length,
    )

    # 8. 训练参数
    training_args = TrainingArguments(
        output_dir=REWITING_OUTPUT,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        lr_scheduler_type=training_config.lr_scheduler_type,
        warmup_ratio=training_config.warmup_ratio,
        weight_decay=training_config.weight_decay,
        fp16=True,  # Unsloth 推荐开启
        logging_dir=f"{REWITING_OUTPUT}/logs",
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        eval_steps=training_config.eval_steps,
        eval_strategy=training_config.eval_strategy,
        save_total_limit=training_config.save_total_limit,
        report_to=training_config.report_to,
        remove_unused_columns=False,
        predict_with_generate=True,
        gradient_checkpointing=True,
    )

    # 9. 使用 SFTTrainer
    from trl import SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        max_seq_length=training_config.rewriting_max_length,
    )

    # 10. 开始训练（只训练 rewritin adapter）
    logger.info("Starting rewriting training (only rewriting LoRA will be updated)...")
    train_result = trainer.train()

    # 11. 保存
    logger.info("Saving models...")

    # 保存改写 LoRA
    model.save_pretrained(REWITING_OUTPUT)
    logger.info(f"[OK] Rewriting LoRA saved to: {REWITING_OUTPUT}")

    # 保存 tokenizer
    tokenizer.save_pretrained(REWITING_OUTPUT)

    logger.info(f"[OK] Done!")
    logger.info(f"  - Rewriting LoRA: {REWITING_OUTPUT}/rewriting")
    logger.info(f"  - Detection LoRA: {DETECTION_CHECKPOINT}")

    return train_result.metrics

# 5. 运行
if __name__ == "__main__":
    metrics = train_rewriting_only()
    print("Training metrics:", metrics)
