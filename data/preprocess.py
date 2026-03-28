"""
数据预处理模块
"""

import json
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path
from datasets import Dataset as HFDataset
from .dataset import save_json_data


# 示例模板（用于生成合成数据）
DETECTION_PROMPT_TEMPLATES = [
    "判断以下文本是否由 AI 生成：{text}",
    "请识别以下内容是否为 AI 生成：{text}",
    "这段文字是 AI 写的还是人写的？{text}",
]

REWRITING_PROMPT_TEMPLATES = [
    "改写以下 AI 生成的文本，使其更自然：{input}\n改写后：",
    "将下面的 AI 生成内容改写得像人类写作：{input}\n改写结果：",
]


def create_synthetic_detection_data(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """
    创建合成的检测数据（用于测试）
    实际项目中应该使用真实的 HC3、SciAI 等数据集
    """
    data = []
    
    # AI 生成的样本（label=1）
    ai_samples = [
        "基于上述实验结果，我们可以得出以下结论：该方法在准确率和效率方面均表现出色，显著优于现有方案。",
        "本研究旨在探讨人工智能在教育领域的应用现状与未来发展趋势，通过文献综述和案例分析相结合的方法。",
        "随着深度学习技术的快速发展，自然语言处理领域取得了显著进展，尤其是在机器翻译和文本生成方面。",
        "本论文提出了一种新颖的卷积神经网络架构，通过引入注意力机制有效提升了图像分类的准确率。",
        "实验结果表明，我们提出的算法在处理大规模数据集时表现出优异的性能和可扩展性。",
    ]
    
    # 人类写作样本（label=0）
    human_samples = [
        "我今天去了超市，买了一些水果和牛奶。天气不错，适合出门。",
        "昨晚看了那部新电影，剧情挺有意思的，推荐你也去看看。",
        "这周的工作任务有点多，但我会尽量按时完成的。大家加油！",
        "刚学会做一道新菜，虽然有点咸，但总体来说还不错。",
        "周末打算去爬山，希望天气好，不然又要改计划了。",
    ]
    
    for i in range(num_samples // 2):
        # AI 样本
        text = random.choice(ai_samples)
        data.append({"text": text, "label": 1})
        
        # 人类样本
        text = random.choice(human_samples)
        data.append({"text": text, "label": 0})
    
    random.shuffle(data)
    return data


def create_synthetic_rewriting_data(num_pairs: int = 1000) -> List[Dict[str, Any]]:
    """
    创建合成的改写数据（用于测试）
    实际项目中应该使用真实的平行语料
    """
    data = []
    
    pairs = [
        (
            "According to the experimental data, we can conclude that this method performs excellently in both accuracy and efficiency, significantly outperforming existing solutions.",
            "实验结果表明，该方法在准确率和效率上均表现优异，显著优于现有方案。"
        ),
        (
            "The rapid development of artificial intelligence technology has brought profound impacts to various industries.",
            "人工智能技术的快速发展给各行各业带来了深远影响。"
        ),
        (
            "In this paper, we propose a novel approach to solve the long-standing problem of efficient data processing.",
            "本文提出了一种新颖的方法，用于解决长期存在的高效数据处理问题。"
        ),
        (
            "The experimental results demonstrate that our algorithm achieves state-of-the-art performance on multiple benchmark datasets.",
            "实验结果表明，我们的算法在多个基准数据集上达到了最先进的性能。"
        ),
        (
            "With the continuous advancement of technology, the application scenarios of machine learning are becoming increasingly widespread.",
            "随着技术的不断进步，机器学习的应用场景越来越广泛。"
        ),
    ]
    
    for i in range(num_pairs):
        input_text, output_text = random.choice(pairs)
        data.append({"input": input_text, "output": output_text})
    
    return data


def split_train_eval(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.9
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    划分训练集和验证集
    """
    split_idx = int(len(data) * train_ratio)
    random.shuffle(data)
    return data[:split_idx], data[split_idx:]


def prepare_dummy_datasets(output_dir: str = "data"):
    """
    准备示例数据集（用于快速测试）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建检测数据
    print("Creating synthetic detection data...")
    detection_data = create_synthetic_detection_data(200)
    train_det, eval_det = split_train_eval(detection_data)
    
    save_json_data(train_det, output_path / "detection" / "train.json")
    save_json_data(eval_det, output_path / "detection" / "eval.json")
    print(f"  Detection: {len(train_det)} train, {len(eval_det)} eval")
    
    # 创建改写数据
    print("Creating synthetic rewriting data...")
    rewriting_data = create_synthetic_rewriting_data(100)
    train_rew, eval_rew = split_train_eval(rewriting_data)
    
    save_json_data(train_rew, output_path / "rewriting" / "train.json")
    save_json_data(eval_rew, output_path / "rewriting" / "eval.json")
    print(f"  Rewriting: {len(train_rew)} train, {len(eval_rew)} eval")
    
    print(f"[OK] Dummy datasets saved to {output_path}")


if __name__ == "__main__":
    prepare_dummy_datasets()
