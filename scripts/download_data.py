#!/usr/bin/env python3
"""
下载并处理 AIGC 检测和改写数据集
支持: HC3 (中文人类 vs ChatGPT 对比数据)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset


def download_hc3():
    """下载 HC3 数据集"""
    print("Downloading HC3 dataset...")
    print("   (May take a few minutes depending on network)")

    # HC3 中文数据集
    dataset = load_dataset("Hello-SimpleAI/HC3", name="CHN", trust_remote_code=True)
    print(f"[OK] HC3 download complete!")
    print(f"   Train: {len(dataset['train'])} samples")
    if 'test' in dataset:
        print(f"   Test: {len(dataset['test'])} samples")
    if 'val' in dataset:
        print(f"   Val: {len(dataset['val'])} samples")

    return dataset


def process_hc3_to_detection(dataset, max_samples: int = 10000) -> List[Dict[str, Any]]:
    """
    将 HC3 转换为检测任务数据
    HC3 每条数据包含: question, human_answers, chatgpt_answers
    """
    print("[PROCESS] Converting to detection data...")

    data = []
    count = 0

    for item in dataset['train']:
        # 人类回答 (label=0)
        for human_answer in item.get('human_answers', []):
            if human_answer and len(human_answer.strip()) > 10:
                data.append({
                    "text": human_answer.strip(),
                    "label": 0
                })
                count += 1
                if count >= max_samples // 2:
                    break

        # ChatGPT 回答 (label=1)
        for gpt_answer in item.get('chatgpt_answers', []):
            if gpt_answer and len(gpt_answer.strip()) > 10:
                data.append({
                    "text": gpt_answer.strip(),
                    "label": 1
                })
                count += 1
                if count >= max_samples:
                    break

        if count >= max_samples:
            break

    random.shuffle(data)
    print(f"   检测数据: {len(data)} 条 (人类: {sum(d['label']==0 for d in data)}, AI: {sum(d['label']==1 for d in data)})")
    return data


def create_rewriting_pairs(detection_data: List[Dict[str, Any]], max_pairs: int = 5000) -> List[Dict[str, Any]]:
    """
    从检测数据中创建改写对
    AI生成文本 -> 改写成人类风格

    注意: HC3 没有直接的改写对，这里只是演示格式
    真实改写对需要自己生成或使用其他数据集
    """
    print("[PROCESS] 正在生成改写对 (模拟)...")
    print("   [WARNING] HC3 has no real rewriting pairs, using simulated data")
    print("   [TIP] Suggest generating real rewriting pairs with GPT-4 later for better results")

    # 找 AI 生成的文本
    ai_texts = [d['text'] for d in detection_data if d['label'] == 1]

    pairs = []
    for text in ai_texts[:min(len(ai_texts), max_pairs)]:
        # 简单的"改写"：保留原意但稍微修改表达
        # 真实场景需要人类写作的版本
        pairs.append({
            "input": text,
            "output": text  # 这里应该放真正的人类写作版本
        })

    print(f"   改写对: {len(pairs)} 对 (格式验证用)")
    return pairs


def split_train_eval(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.9
) -> tuple:
    """划分训练集和验证集"""
    split_idx = int(len(data) * train_ratio)
    random.shuffle(data)
    return data[:split_idx], data[split_idx:]


def save_json_data(data: List[Dict[str, Any]], filepath: Path):
    """保存为 JSON 格式"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"   [OK] Saved: {filepath}")


def main():
    print("=" * 60)
    print("[Yety] Dataset Download Script")
    print("=" * 60)

    # 创建输出目录
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    detection_dir = data_dir / "detection"
    rewriting_dir = data_dir / "rewriting"

    # 1. 下载 HC3 数据集
    dataset = download_hc3()

    # 2. 转换为检测数据
    detection_data = process_hc3_to_detection(dataset, max_samples=10000)

    # 3. 创建改写对 (模拟)
    rewriting_data = create_rewriting_pairs(detection_data, max_pairs=5000)

    # 4. 划分训练集和验证集
    det_train, det_eval = split_train_eval(detection_data)
    rew_train, rew_eval = split_train_eval(rewriting_data)

    # 5. 保存数据
    print("\n[SAVE] Saving data...")
    save_json_data(det_train, detection_dir / "train.json")
    save_json_data(det_eval, detection_dir / "eval.json")
    save_json_data(rew_train, rewriting_dir / "train.json")
    save_json_data(rew_eval, rewriting_dir / "eval.json")

    print("\n" + "=" * 60)
    print("[OK] Dataset preparation complete!")
    print("=" * 60)
    print(f"\nData directory:")
    print(f"   Detection: {detection_dir}")
    print(f"   Rewriting: {rewriting_dir}")
    print(f"\nData stats:")
    print(f"   Detection train: {len(det_train)} samples")
    print(f"   Detection eval: {len(det_eval)} samples")
    print(f"   Rewriting train: {len(rew_train)} pairs")
    print(f"   Rewriting eval: {len(rew_eval)} pairs")
    print(f"\nNext step: python main.py train")


if __name__ == "__main__":
    main()
