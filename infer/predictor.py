"""
Yety 推理接口 - 封装检测和改写功能
"""

import torch
from transformers import AutoTokenizer
from typing import Dict, Any, List
import os

from models.multitask_model import MultiTaskModel
from config.model_config import ModelConfig


class YetyPredictor:
    """
    Yety 项目的推理接口
    
    功能：
    - 检测文本是否 AI 生成（二分类）
    - 将 AI 生成文本改写成更自然的表达
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        base_model_name: str = None,
        device_map: str = "auto"
    ):
        """
        初始化预测器
        
        Args:
            checkpoint_dir: 检查点目录（包含 detection/ 和 rewriting/ 子目录）
            base_model_name: 基础模型名称（可选，默认从配置读取）
            device_map: 设备映射策略
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device_map = device_map
        
        # 加载配置
        self.model_config = ModelConfig()
        if base_model_name:
            self.model_config.base_model_name = base_model_name
        
        # 初始化多任务模型管理器
        print("[START] Initializing YetyPredictor...")
        self.model_manager = MultiTaskModel(
            base_model_name=self.model_config.base_model_name,
            model_config=self.model_config,
            device_map=device_map
        )
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model_name,
            trust_remote_code=self.model_config.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载 adapter
        self._load_adapters()
        
        print("[OK] YetyPredictor ready!")
    
    def _load_adapters(self):
        """加载检测和改写的 LoRA adapter"""
        detection_path = self.checkpoint_dir / "detection"
        rewriting_path = self.checkpoint_dir / "rewriting"
        
        if detection_path.exists():
            print(f"Loading detection adapter from {detection_path}")
            self.model_manager.load_adapter(
                "detection",
                str(detection_path),
                task_type="cls"
            )
        else:
            raise FileNotFoundError(f"Detection adapter not found at {detection_path}")
        
        if rewriting_path.exists():
            print(f"Loading rewriting adapter from {rewriting_path}")
            self.model_manager.load_adapter(
                "rewriting",
                str(rewriting_path),
                task_type="causal_lm"
            )
        else:
            raise FileNotFoundError(f"Rewriting adapter not found at {rewriting_path}")
    
    def detect(self, text: str, return_probs: bool = False) -> Dict[str, Any]:
        """
        检测文本是否由 AI 生成
        
        Args:
            text: 待检测的文本
            return_probs: 是否返回概率分布
            
        Returns:
            包含检测结果和置信度的字典
        """
        # 切换到检测 adapter
        model = self.model_manager.set_active_adapter("detection")
        model.eval()
        
        # 构建提示词
        prompt = f"判断以下文本是否由 AI 生成：{text}"
        
        # 分词
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 移动到模型设备
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # 推理
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # 获取 [CLS] 位置的 hidden state
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]  # 取最后一层
                cls_token = hidden_states[:, 0, :]
            else:
                # fallback: 使用 logits 的第一行
                logits = outputs.logits
                cls_token = logits[:, 0, :]
            
            # 分类头
            logits = model.classifier(cls_token)
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
        
        result = {
            "label": "AI生成" if pred == 1 else "人类写作",
            "confidence": probs[0][pred].item()
        }
        
        if return_probs:
            result["probabilities"] = {
                "human": probs[0][0].item(),
                "ai": probs[0][1].item()
            }
        
        return result
    
    def rewrite(
        self,
        text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        改写文本
        
        Args:
            text: 待改写的 AI 生成文本
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            do_sample: 是否使用采样
            **generation_kwargs: 其他生成参数
            
        Returns:
            包含改写结果的字典
        """
        # 切换到改写 adapter
        model = self.model_manager.set_active_adapter("rewriting")
        model.eval()
        
        # 构建提示词
        prompt = f"改写以下 AI 生成的文本，使其更自然：{text}\n改写后："
        
        # 分词
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # 移动到模型设备
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **generation_kwargs
            )
        
        # 只取生成部分
        generated_tokens = outputs[0][input_ids.shape[1]:]
        rewritten = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {
            "original_text": text,
            "rewritten_text": rewritten.strip(),
            "generated_tokens": len(generated_tokens)
        }
    
    def batch_detect(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """批量检测"""
        results = []
        for text in texts:
            result = self.detect(text, **kwargs)
            results.append(result)
        return results
    
    def batch_rewrite(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """批量改写"""
        results = []
        for text in texts:
            result = self.rewrite(text, **kwargs)
            results.append(result)
        return results
    
    def set_device_map(self, device_map: str):
        """动态设置设备映射（需要重新加载模型）"""
        self.model_manager = MultiTaskModel(
            base_model_name=self.model_config.base_model_name,
            model_config=self.model_config,
            device_map=device_map
        )
        # 重新加载 adapter
        self._load_adapters()


def test_predictor():
    """测试预测器功能"""
    # 这是一个测试函数，需要先训练好模型才能运行
    
    predictor = YetyPredictor(
        checkpoint_dir="checkpoints/joint_iter0",
        base_model_name="Qwen/Qwen3.5-2B"
    )
    
    # 测试检测
    test_text = "基于上述实验结果，我们可以得出以下结论：该方法在准确率和效率方面均表现出色。"
    detection_result = predictor.detect(test_text, return_probs=True)
    print(f"Detection result: {detection_result}")
    
    # 测试改写
    rewrite_result = predictor.rewrite(test_text)
    print(f"Rewrite result: {rewrite_result}")


if __name__ == "__main__":
    print("This is a module. Import and use YetyPredictor class.")
    print("To test, run: from infer.predictor import test_predictor")
