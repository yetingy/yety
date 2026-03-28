"""
模型和 LoRA 配置
"""

import os
from dataclasses import dataclass, field
from peft import LoraConfig, TaskType
import torch


@dataclass
class ModelConfig:
    """模型和 LoRA 配置类"""

    # 基础模型 - 使用 ModelScope 镜像
    base_model_name: str = "Qwen/Qwen3.5-2B"
    trust_remote_code: bool = True

    # ModelScope 配置
    use_modelscope: bool = True
    modelscope_model_id: str = "qwen/Qwen3.5-2B"

    # 量化配置
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # 检测任务 LoRA 配置
    detection_lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # 分类任务
        r=8,                      # rank - 分类任务相对简单，rank 可以小一些
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        modules_to_save=["classifier"],  # 额外添加的分类头需要保存
    )

    # 改写任务 LoRA 配置
    rewriting_lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 生成任务
        r=16,                     # 生成任务需要更大的 capacity
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    # 分类头配置（检测任务）
    num_labels: int = 2  # AI生成 / 人类写作

    # 设备配置
    device_map: str = "auto"

    def get_model_path(self):
        """获取模型路径，优先使用 ModelScope 下载"""
        if self.use_modelscope:
            from modelscope import snapshot_download
            cache_dir = os.path.expanduser("~/.cache/modelscope/models")
            print(f"[ModelScope] Downloading model to {cache_dir}...")
            model_dir = snapshot_download(
                self.modelscope_model_id,
                cache_dir=cache_dir
            )
            print(f"[ModelScope] Model downloaded to: {model_dir}")
            return model_dir
        return self.base_model_name

    def get_quantization_config(self):
        """获取 bitsandbytes 量化配置"""
        from transformers import BitsAndBytesConfig

        compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)

        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        )
