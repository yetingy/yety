"""
Checkpoint 管理工具
"""

import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class CheckpointManager:
    """
    Checkpoint 管理器
    
    功能：
    1. 保存训练检查点（模型 + 优化器 + 调度器状态）
    2. 加载检查点
    3. 管理多个检查点（保留最佳 N 个）
    4. 记录训练历史
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: int = 3
    ):
        """
        初始化
        
        Args:
            checkpoint_dir: 检查点目录
            max_to_keep: 最多保留的检查点数量
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        
        # 历史记录文件
        self.history_file = self.checkpoint_dir / "checkpoint_history.json"
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """加载检查点历史"""
        if self.history_file.exists():
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _save_history(self):
        """保存检查点历史"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def save(
        self,
        model,
        optimizer,
        scheduler=None,
        epoch: int = 0,
        global_step: int = 0,
        metrics: Dict[str, float] = None,
        extra_state: Dict[str, Any] = None,
        checkpoint_name: str = None
    ) -> str:
        """
        保存检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器（可选）
            epoch: 当前 epoch
            global_step: 当前 global step
            metrics: 评估指标
            extra_state: 额外状态
            checkpoint_name: 检查点名称（可选）
            
        Returns:
            检查点路径
        """
        if checkpoint_name is None:
            checkpoint_name = f"epoch-{epoch:04d}_step-{global_step:06d}"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型状态
        if hasattr(model, 'save_pretrained'):
            # PeftModel 使用 save_pretrained
            model.save_pretrained(checkpoint_path / "model")
        else:
            torch.save(
                model.state_dict(),
                checkpoint_path / "model.pt"
            )
        
        # 保存优化器状态
        torch.save(
            optimizer.state_dict(),
            checkpoint_path / "optimizer.pt"
        )
        
        # 保存调度器状态
        if scheduler is not None:
            torch.save(
                scheduler.state_dict(),
                checkpoint_path / "scheduler.pt"
            )
        
        # 保存训练状态
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {},
            "extra_state": extra_state or {}
        }
        
        with open(checkpoint_path / "training_state.json", 'w') as f:
            json.dump(state, f, indent=2)
        
        # 更新历史记录
        self.history.append({
            "name": checkpoint_name,
            "path": str(checkpoint_path),
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics or {},
            "timestamp": state["timestamp"]
        })
        self._save_history()
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
        
        print(f"[OK] Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_latest(
        self,
        model,
        optimizer=None,
        scheduler=None,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        加载最新的检查点
        
        Args:
            model: 模型
            optimizer: 优化器（可选）
            scheduler: 调度器（可选）
            device: 设备
            
        Returns:
            训练状态字典
        """
        if not self.history:
            print("No checkpoints found. Starting from scratch.")
            return {}
        
        latest = self.history[-1]
        checkpoint_path = Path(latest["path"])
        
        return self.load(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            device
        )
    
    def load(
        self,
        checkpoint_path: str,
        model,
        optimizer=None,
        scheduler=None,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        加载指定检查点
        
        Args:
            checkpoint_path: 检查点路径
            model: 模型
            optimizer: 优化器（可选）
            scheduler: 调度器（可选）
            device: 设备
            
        Returns:
            训练状态字典
        """
        checkpoint_path = Path(checkpoint_path)
        
        # 加载模型
        model_path = checkpoint_path / "model"
        if model_path.exists():
            if hasattr(model, 'load_pretrained'):
                model = model.from_pretrained(model_path)
            else:
                state_dict = torch.load(
                    model_path / "pytorch_model.bin",
                    map_location=device
                )
                model.load_state_dict(state_dict)
        else:
            # 兼容旧格式
            model_path = checkpoint_path / "model.pt"
            if model_path.exists():
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
        
        # 加载优化器
        if optimizer is not None:
            opt_path = checkpoint_path / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(
                    torch.load(opt_path, map_location=device)
                )
        
        # 加载调度器
        if scheduler is not None:
            sched_path = checkpoint_path / "scheduler.pt"
            if sched_path.exists():
                scheduler.load_state_dict(
                    torch.load(sched_path, map_location=device)
                )
        
        # 加载训练状态
        state_file = checkpoint_path / "training_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
        else:
            state = {}
        
        print(f"[OK] Checkpoint loaded from {checkpoint_path}")
        return state
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点，只保留最新的 max_to_keep 个"""
        if len(self.history) > self.max_to_keep:
            # 保留最新的 max_to_keep 个
            to_delete = self.history[:-self.max_to_keep]
            
            for item in to_delete:
                checkpoint_path = Path(item["path"])
                if checkpoint_path.exists():
                    # 删除目录下的所有文件
                    for file in checkpoint_path.glob("*"):
                        file.unlink()
                    checkpoint_path.rmdir()
                    print(f"🗑️  Deleted old checkpoint: {checkpoint_path}")
            
            # 更新历史记录
            self.history = self.history[-self.max_to_keep:]
            self._save_history()
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """列出所有检查点"""
        return self.history.copy()
    
    def get_best_checkpoint(
        self,
        metric_name: str = "eval_f1",
        higher_is_better: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        获取最佳检查点（基于指定指标）
        
        Args:
            metric_name: 指标名称
            higher_is_better: 是否越大越好
            
        Returns:
            最佳检查点信息
        """
        if not self.history:
            return None
        
        best_idx = 0
        best_value = self.history[0]["metrics"].get(metric_name)
        
        if best_value is None:
            return None
        
        for i, item in enumerate(self.history[1:], start=1):
            value = item["metrics"].get(metric_name)
            if value is None:
                continue
            
            if higher_is_better and value > best_value:
                best_value = value
                best_idx = i
            elif not higher_is_better and value < best_value:
                best_value = value
                best_idx = i
        
        return self.history[best_idx]


if __name__ == "__main__":
    # 测试 CheckpointManager
    manager = CheckpointManager("./test_checkpoints", max_to_keep=2)
    
    # 模拟保存几个检查点
    for i in range(3):
        manager.save(
            model=None,  # 测试用 None
            optimizer=None,
            epoch=i,
            global_step=i * 100,
            metrics={"accuracy": 0.8 + i * 0.05}
        )
    
    print("Checkpoints:", manager.list_checkpoints())
    best = manager.get_best_checkpoint("accuracy", higher_is_better=True)
    print(f"Best checkpoint: {best}")
