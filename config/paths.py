"""
路径管理配置
"""

from pathlib import Path
import os


class PathManager:
    """统一路径管理"""
    
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        self.checkpoints_dir = self.project_root / "checkpoints"
        self.logs_dir = self.project_root / "logs"
        self.data_dir = self.project_root / "data"
        self.config_dir = self.project_root / "config"
        self.models_dir = self.project_root / "models"
        self.train_dir = self.project_root / "train"
        self.eval_dir = self.project_root / "eval"
        self.infer_dir = self.project_root / "infer"
        self.utils_dir = self.project_root / "utils"
        self.scripts_dir = self.project_root / "scripts"
        
        # 确保目录存在
        self._create_dirs()
    
    def _create_dirs(self):
        """创建必要的目录"""
        dirs = [
            self.checkpoints_dir,
            self.logs_dir,
            self.data_dir,
            self.config_dir,
            self.models_dir,
            self.train_dir,
            self.eval_dir,
            self.infer_dir,
            self.utils_dir,
            self.scripts_dir,
            self.data_dir / "detection",
            self.data_dir / "rewriting",
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_checkpoint_path(self, task, iteration=None):
        """获取 checkpoint 路径"""
        if iteration is not None:
            return self.checkpoints_dir / f"{task}_iter{iteration}"
        return self.checkpoints_dir / task
    
    def get_log_path(self, task, iteration=None):
        """获取日志路径"""
        if iteration is not None:
            return self.logs_dir / f"{task}_iter{iteration}.log"
        return self.logs_dir / f"{task}.log"
