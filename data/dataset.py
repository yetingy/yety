"""
数据集类定义
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Any
import json
import os


class DetectionDataset(Dataset):
    """AIGC 检测数据集 - 分类任务"""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        prompt_template: str = "判断以下文本是否由 AI 生成：{text}"
    ):
        """
        Args:
            data: 数据列表，每个元素包含 "text" 和 "label"
            tokenizer: 分词器
            max_length: 最大序列长度
            prompt_template: 提示词模板
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        
        # 确保 tokenizer 有 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item["text"]
        label = item["label"]
        
        # 构建提示词
        prompt = self.prompt_template.format(text=text)
        
        # 分词
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class RewritingDataset(Dataset):
    """文本改写数据集 - seq2seq 生成任务"""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        input_template: str = "改写以下 AI 生成的文本，使其更自然：{input}\n改写后：",
        max_new_tokens: int = 256
    ):
        """
        Args:
            data: 数据列表，每个元素包含 "input" 和 "output"
            tokenizer: 分词器
            max_length: 最大序列长度
            input_template: 输入提示词模板
            max_new_tokens: 最大生成 token 数（用于截断输出）
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_template = input_template
        self.max_new_tokens = max_new_tokens
        
        # 确保 tokenizer 有 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        input_text = item["input"]
        output_text = item["output"]
        
        # 构建完整文本
        full_text = self.input_template.format(input=input_text) + output_text
        
        # 分词
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 计算输入部分的长度，将输入部分的 label 设为 -100
        prompt = self.input_template.format(input=input_text)
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        prompt_len = len(prompt_tokens["input_ids"][0])
        
        labels = encoding["input_ids"].clone()
        labels[0, :prompt_len] = -100  # 只计算输出部分的 loss
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }


def load_json_data(filepath: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """加载 JSON 格式的数据集"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples and len(data) > max_samples:
        data = data[:max_samples]
    
    return data


def save_json_data(data: List[Dict[str, Any]], filepath: str):
    """保存数据集为 JSON 格式"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
