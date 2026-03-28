#!/usr/bin/env python3
"""
Yety - AIGC 检测与改写工具
主入口脚本

使用方法：
  python main.py train --iterations 3
  python main.py detect --text "你的文本"
  python main.py rewrite --text "你的文本"
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from models.multitask_model import MultiTaskModel
from infer.predictor import YetyPredictor
from data.preprocess import prepare_dummy_datasets
import logging


def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def cmd_train(args):
    """训练命令"""
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("Yety 双 LoRA 训练启动")
    logger.info("="*60)
    
    # 1. 准备数据
    if args.prepare_data:
        logger.info("准备示例数据集...")
        prepare_dummy_datasets()
    
    # 2. 加载配置
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # 覆盖配置
    if args.iterations:
        training_config.alternating_iterations = args.iterations
    if args.epochs:
        training_config.num_train_epochs = args.epochs
    if args.batch_size:
        training_config.per_device_train_batch_size = args.batch_size
    
    # 3. 初始化模型管理器
    logger.info(f"加载基础模型: {model_config.base_model_name}")
    model_manager = MultiTaskModel(
        base_model_name=model_config.base_model_name,
        model_config=model_config,
        device_map=model_config.device_map
    )
    
    # 4. 准备数据路径
    detection_data = {
        "train": "data/detection/train.json",
        "eval": "data/detection/eval.json"
    }
    rewriting_data = {
        "train": "data/rewriting/train.json",
        "eval": "data/rewriting/eval.json"
    }
    
    # 5. 导入并运行交替训练
    from train.train_alternating import train_alternating
    
    history = train_alternating(
        model_manager=model_manager,
        detection_data=detection_data,
        rewriting_data=rewriting_data,
        training_config=training_config,
        model_config=model_config,
        iterations=training_config.alternating_iterations
    )
    
    logger.info("[OK] 训练完成！")
    logger.info(f"检查点保存在: {training_config.output_dir}/")
    
    return history


def cmd_detect(args):
    """检测命令"""
    logger = setup_logging()
    
    checkpoint_dir = args.checkpoint_dir or "checkpoints/joint_iter0"
    
    logger.info(f"加载检查点: {checkpoint_dir}")
    predictor = YetyPredictor(
        checkpoint_dir=checkpoint_dir,
        base_model_name=args.model
    )
    
    result = predictor.detect(
        args.text,
        return_probs=True
    )
    
    print("\n" + "="*40)
    print("检测结果:")
    print(f"  文本: {args.text[:100]}...")
    print(f"  标签: {result['label']}")
    print(f"  置信度: {result['confidence']:.2%}")
    if "probabilities" in result:
        print(f"  概率分布: {result['probabilities']}")
    print("="*40)
    
    return result


def cmd_rewrite(args):
    """改写命令"""
    logger = setup_logging()
    
    checkpoint_dir = args.checkpoint_dir or "checkpoints/joint_iter0"
    
    logger.info(f"加载检查点: {checkpoint_dir}")
    predictor = YetyPredictor(
        checkpoint_dir=checkpoint_dir,
        base_model_name=args.model
    )
    
    result = predictor.rewrite(
        args.text,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    print("\n" + "="*40)
    print("改写结果:")
    print(f"  原文: {args.text[:100]}...")
    print(f"  改写: {result['rewritten_text']}")
    print(f"  生成 token 数: {result['generated_tokens']}")
    print("="*40)
    
    return result


def cmd_test(args):
    """测试命令 - 验证框架"""
    logger = setup_logging()
    
    logger.info("运行框架测试...")
    
    # 1. 测试配置
    logger.info("1. 测试配置加载...")
    model_config = ModelConfig()
    training_config = TrainingConfig()
    assert model_config.base_model_name is not None
    assert training_config.num_train_epochs > 0
    logger.info("[OK] 配置加载成功")
    
    # 2. 测试 MultiTaskModel 初始化（不加载完整模型，只测试结构）
    logger.info("2. 测试 MultiTaskModel 结构...")
    try:
        # 这里只测试类是否可以实例化（不实际加载模型）
        from models.multitask_model import MultiTaskModel
        logger.info("[OK] MultiTaskModel 类导入成功")
    except Exception as e:
        logger.error(f"[ERROR] MultiTaskModel 类导入失败: {e}")
        return False
    
    # 3. 测试数据集类
    logger.info("3. 测试数据集类...")
    try:
        from data.dataset import DetectionDataset, RewritingDataset
        logger.info("[OK] 数据集类导入成功")
    except Exception as e:
        logger.error(f"[ERROR] 数据集类导入失败: {e}")
        return False
    
    # 4. 测试路径管理
    logger.info("4. 测试路径管理...")
    try:
        from config.paths import PathManager
        pm = PathManager()
        assert pm.project_root.exists()
        logger.info("[OK] 路径管理成功")
    except Exception as e:
        logger.error(f"[ERROR] 路径管理失败: {e}")
        return False
    
    logger.info("="*60)
    logger.info("[OK] 所有框架测试通过！")
    logger.info("可以开始训练了：python main.py train")
    logger.info("="*60)
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Yety - AIGC Detection and Rewriting with Dual LoRA"
    )
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # train 命令
    parser_train = subparsers.add_parser("train", help="训练双 LoRA 模型")
    parser_train.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="交替训练迭代次数"
    )
    parser_train.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="每个任务的训练轮数"
    )
    parser_train.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批次大小"
    )
    parser_train.add_argument(
        "--prepare-data",
        action="store_true",
        help="是否准备示例数据"
    )
    parser_train.set_defaults(func=cmd_train)
    
    # detect 命令
    parser_detect = subparsers.add_parser("detect", help="检测文本")
    parser_detect.add_argument(
        "text",
        type=str,
        help="待检测的文本"
    )
    parser_detect.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="检查点目录"
    )
    parser_detect.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-2B",
        help="基础模型名称"
    )
    parser_detect.set_defaults(func=cmd_detect)
    
    # rewrite 命令
    parser_rewrite = subparsers.add_parser("rewrite", help="改写文本")
    parser_rewrite.add_argument(
        "text",
        type=str,
        help="待改写的文本"
    )
    parser_rewrite.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="检查点目录"
    )
    parser_rewrite.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-2B",
        help="基础模型名称"
    )
    parser_rewrite.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="最大生成 token 数"
    )
    parser_rewrite.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度"
    )
    parser_rewrite.set_defaults(func=cmd_rewrite)
    
    # test 命令
    parser_test = subparsers.add_parser("test", help="测试框架")
    parser_test.set_defaults(func=cmd_test)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except Exception as e:
        logger = setup_logging()
        logger.error(f"命令执行失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
