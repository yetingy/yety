"""
多任务模型管理器 - 支持双 LoRA adapter
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from peft import PeftModel, PeftConfig, TaskType, get_peft_model
from typing import Dict, Optional
import warnings


class MultiTaskModel:
    """
    支持双 LoRA 的多任务模型管理器
    
    功能：
    1. 加载基础模型（支持 4bit 量化）
    2. 为不同任务添加独立的 LoRA adapter
    3. 支持动态切换活跃 adapter
    4. 分别保存/加载每个任务的 adapter
    """
    
    def __init__(
        self,
        base_model_name: str,
        model_config,
        device_map: str = "auto"
    ):
        """
        初始化多任务模型
        
        Args:
            base_model_name: 基础模型名称或路径
            model_config: ModelConfig 实例，包含 LoRA 配置
            device_map: 设备映射策略
        """
        self.base_model_name = base_model_name
        self.model_config = model_config
        self.device_map = device_map

        # 存储 adapter 模型
        self._adapters: Dict[str, PeftModel] = {}
        self.active_adapter: Optional[str] = None

        # 加载基础模型
        self.model_path = self.model_config.get_model_path()
        self._load_base_model()

        # 添加分类头（用于检测任务）
        self._init_classifier()
    
    def _load_base_model(self):
        """加载基础模型（支持量化）"""
        print(f"[INFO] Loading base model from: {self.model_path}")

        # 量化配置
        bnb_config = self.model_config.get_quantization_config()

        # 加载模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map=self.device_map,
            trust_remote_code=self.model_config.trust_remote_code
        )

        # 确保基础模型有 pad_token
        if self.base_model.config.pad_token_id is None:
            self.base_model.config.pad_token_id = self.base_model.config.eos_token_id

        print(f"[OK] Base model loaded (device_map: {self.device_map})")
    
    def _init_classifier(self):
        """初始化分类头（用于检测任务）"""
        hidden_size = self.base_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, self.model_config.num_labels)

        # 移动到正确的设备并使用与模型相同的 dtype
        if hasattr(self.base_model, 'device'):
            device = self.base_model.device
        elif hasattr(self.base_model, 'device_map'):
            # 如果使用 device_map，需要手动处理
            device = next(self.base_model.parameters()).device
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 获取模型参数的 dtype
        model_dtype = next(self.base_model.parameters()).dtype
        self.classifier = self.classifier.to(device=device, dtype=model_dtype)

        print(f"[OK] Classifier head initialized (hidden_size={hidden_size}, num_labels={self.model_config.num_labels}, dtype={model_dtype})")
    
    def add_adapter(
        self,
        adapter_name: str,
        lora_config,
        task_type: str = "cls"
    ) -> PeftModel:
        """
        添加 LoRA adapter

        Args:
            adapter_name: adapter 名称（如 "detection", "rewriting"）
            lora_config: LoraConfig 对象
            task_type: 任务类型，"cls" 或 "causal_lm"

        Returns:
            添加了 adapter 的模型
        """
        if adapter_name in self._adapters:
            warnings.warn(f"Adapter '{adapter_name}' already exists, returning existing one")
            return self._adapters[adapter_name]

        print(f"Adding adapter: {adapter_name} (task_type: {task_type})")

        # 如果 base_model 还不是 PeftModel，先用 get_peft_model 包装
        if not isinstance(self.base_model, PeftModel):
            # 使用 get_peft_model 创建 PeftModel 并添加第一个 adapter
            self.base_model = get_peft_model(
                self.base_model,
                lora_config,
                adapter_name=adapter_name
            )
            model = self.base_model
        else:
            # 已经是 PeftModel，直接添加 adapter
            model = self.base_model.add_adapter(lora_config, adapter_name)

        if task_type == "cls":
            # 分类任务：需要额外的分类头
            model.classifier = self.classifier

        self._adapters[adapter_name] = model
        print(f"[OK] Adapter '{adapter_name}' added successfully")

        return model
    
    def set_active_adapter(self, adapter_name: str) -> PeftModel:
        """
        设置活跃的 adapter
        
        Args:
            adapter_name: adapter 名称
            
        Returns:
            当前活跃的模型
        """
        if adapter_name not in self._adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found. Available: {list(self._adapters.keys())}")
        
        self.active_adapter = adapter_name
        return self._adapters[adapter_name]
    
    def get_active_model(self) -> Optional[PeftModel]:
        """获取当前活跃的模型"""
        if self.active_adapter is None:
            return None
        return self._adapters[self.active_adapter]
    
    def save_adapter(self, adapter_name: str, save_path: str):
        """
        保存单个 adapter
        
        Args:
            adapter_name: adapter 名称
            save_path: 保存路径
        """
        if adapter_name not in self._adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")
        
        model = self._adapters[adapter_name]
        
        # 保存 adapter
        model.save_pretrained(save_path)
        
        # 如果是检测任务，额外保存分类头
        if hasattr(model, 'classifier'):
            classifier_path = f"{save_path}/classifier.pt"
            torch.save(model.classifier.state_dict(), classifier_path)
            print(f"[OK] Classifier head saved to {classifier_path}")
        
        print(f"[OK] Adapter '{adapter_name}' saved to {save_path}")
    
    def load_adapter(
        self,
        adapter_name: str,
        load_path: str,
        task_type: str = "cls"
    ) -> PeftModel:
        """
        加载已保存的 adapter
        
        Args:
            adapter_name: adapter 名称
            load_path: 加载路径
            task_type: 任务类型
            
        Returns:
            加载的模型
        """
        print(f"Loading adapter: {adapter_name} from {load_path}")
        
        if task_type == "cls":
            model = PeftModel.from_pretrained(
                self.base_model,
                load_path,
                adapter_name=adapter_name,
                is_trainable=False
            )
            # 加载分类头
            classifier_path = f"{load_path}/classifier.pt"
            if os.path.exists(classifier_path):
                classifier_state = torch.load(classifier_path, map_location='cpu')
                model.classifier.load_state_dict(classifier_state)
                print(f"[OK] Classifier head loaded from {classifier_path}")
        else:
            model = PeftModel.from_pretrained(
                self.base_model,
                load_path,
                adapter_name=adapter_name,
                is_trainable=False
            )
        
        self._adapters[adapter_name] = model
        print(f"[OK] Adapter '{adapter_name}' loaded successfully")
        
        return model
    
    def list_adapters(self) -> list:
        """列出所有已添加的 adapter"""
        return list(self._adapters.keys())
    
    def set_training_mode(self, mode: bool = True):
        """设置所有 adapter 为训练/评估模式"""
        for adapter in self._adapters.values():
            adapter.train(mode)
