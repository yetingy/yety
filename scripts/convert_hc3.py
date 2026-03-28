#!/usr/bin/env python3
"""
将 HC3 all.jsonl 转换为 Yety 项目的数据格式
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(filepath: str) -> List[Dict]:
    """加载 jsonl 文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def convert_to_detection(data: List[Dict], max_samples: int = 10000) -> List[Dict[str, Any]]:
    """转换为检测数据"""
    detection_data = []
    count = 0

    for item in data:
        # 人类回答 (label=0)
        for answer in item.get('human_answers', []):
            if answer and len(answer.strip()) > 20:
                detection_data.append({
                    "text": answer.strip(),
                    "label": 0
                })
                count += 1
                if count >= max_samples // 2:
                    break

        # ChatGPT 回答 (label=1)
        for answer in item.get('chatgpt_answers', []):
            if answer and len(answer.strip()) > 20:
                detection_data.append({
                    "text": answer.strip(),
                    "label": 1
                })
                count += 1
                if count >= max_samples:
                    break

        if count >= max_samples:
            break

    random.shuffle(detection_data)
    return detection_data


def create_rewriting_pairs(detection_data: List[Dict], max_pairs: int = 3000) -> List[Dict[str, Any]]:
    """
    创建改写对
    这里用人类文本作为 input，ChatGPT 文本作为 output
    实际使用时可能需要反过来或者用更好的配对
    """
    # 找 AI 生成的文本
    ai_texts = [d['text'] for d in detection_data if d['label'] == 1]
    human_texts = [d['text'] for d in detection_data if d['label'] == 0]

    pairs = []
    n = min(len(ai_texts), len(human_texts), max_pairs)

    for i in range(n):
        # AI 生成的 -> 改写成人类风格（用对应的人类回答作为参考）
        pairs.append({
            "input": ai_texts[i],
            "output": human_texts[i] if i < len(human_texts) else ai_texts[i]
        })

    return pairs


def split_train_eval(data: List, train_ratio: float = 0.9):
    """划分训练集和验证集"""
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def save_json(data: List, filepath: Path):
    """保存 JSON"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved: {filepath} ({len(data)} samples)")


def main():
    print("=" * 60)
    print("[Yety] HC3 Data Conversion Script")
    print("=" * 60)

    # 文件路径
    project_root = Path(__file__).parent.parent
    input_file = project_root / "all.jsonl"
    detection_dir = project_root / "data" / "detection"
    rewriting_dir = project_root / "data" / "rewriting"

    if not input_file.exists():
        print(f"[ERROR] File not found: {input_file}")
        return

    # 1. 加载数据
    print(f"[LOAD] Loading {input_file}...")
    raw_data = load_jsonl(str(input_file))
    print(f"       Loaded {len(raw_data)} records")

    # 2. 转换为检测数据
    print("[PROCESS] Converting to detection format...")
    detection_data = convert_to_detection(raw_data, max_samples=10000)
    print(f"       Detection data: {len(detection_data)} samples")
    print(f"       Human (0): {sum(d['label']==0 for d in detection_data)}")
    print(f"       AI (1): {sum(d['label']==1 for d in detection_data)}")

    # 3. 创建改写对
    print("[PROCESS] Creating rewriting pairs...")
    rewriting_data = create_rewriting_pairs(detection_data, max_pairs=3000)
    print(f"       Rewriting pairs: {len(rewriting_data)}")

    # 4. 划分数据集
    det_train, det_eval = split_train_eval(detection_data)
    rew_train, rew_eval = split_train_eval(rewriting_data)

    # 5. 保存
    print("[SAVE] Saving datasets...")
    save_json(det_train, detection_dir / "train.json")
    save_json(det_eval, detection_dir / "eval.json")
    save_json(rew_train, rewriting_dir / "train.json")
    save_json(rew_eval, rewriting_dir / "eval.json")

    print("\n" + "=" * 60)
    print("[OK] Conversion complete!")
    print("=" * 60)
    print(f"\nData stats:")
    print(f"   Detection train: {len(det_train)} | eval: {len(det_eval)}")
    print(f"   Rewriting train: {len(rew_train)} | eval: {len(rew_eval)}")
    print(f"\nNext step: python main.py train")


if __name__ == "__main__":
    main()
