# Yety - AIGC 检测与改写工具

基于 **双 LoRA 架构** 的学术论文 AI 生成内容检测与智能改写工具。

- 🔍 **检测任务**: 识别论文中 AI 生成的内容（二分类）
- ✏️ **改写任务**: 将 AI 生成内容改写成更自然的学术表达
- 🎯 **双 LoRA**: 使用同一个基础模型（Qwen3.5-2B），两个独立的任务 adapter

## ✨ 特性

- **资源共享**: 检测和改写共享基础模型参数，节省显存
- **任务独立**: 每个任务有自己的 LoRA adapter，互不干扰
- **交替训练**: 支持两个任务轮流训练，避免灾难性遗忘
- **灵活部署**: 推理时可动态切换任务
- **轻量化**: 4bit 量化，4GB 显存即可运行

## 📦 项目结构

```
yety/
├── config/              # 配置模块
│   ├── model_config.py   # 模型和 LoRA 配置
│   ├── training_config.py # 训练超参数
│   └── paths.py          # 路径管理
├── data/                # 数据处理
│   ├── dataset.py        # Dataset 类
│   └── preprocess.py     # 数据预处理
├── models/              # 模型定义
│   └── multitask_model.py # 多任务模型管理器
├── train/               # 训练脚本
│   ├── train_detection.py   # 检测任务训练
│   ├── train_rewriting.py   # 改写任务训练
│   ├── train_alternating.py # 交替训练
│   └── trainer.py           # 自定义 Trainer
├── eval/                # 评估模块
│   └── metrics.py        # 评估指标（BLEU, ROUGE, F1, AUC）
├── infer/               # 推理接口
│   └── predictor.py     # YetyPredictor 类
├── utils/               # 工具函数
│   ├── logging.py       # 日志配置
│   └── checkpoint.py    # Checkpoint 管理
├── scripts/             # 脚本
├── main.py              # 主入口
├── requirements.txt     # 依赖
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd yety
pip install -r requirements.txt
```

### 2. 准备数据

```bash
# 生成示例数据（测试用）
python main.py test
# 或手动准备数据，格式见下方
```

### 3. 训练模型

```bash
# 交替训练（默认 3 轮）
python main.py train --prepare-data

# 自定义迭代次数
python main.py train --iterations 5 --epochs 3 --batch-size 4
```

训练后的检查点保存在 `checkpoints/` 目录：
- `checkpoints/detection_iterN/` - 检测 adapter
- `checkpoints/rewriting_iterN/` - 改写 adapter
- `checkpoints/joint_iterN/` - 联合检查点

### 4. 使用模型

```bash
# 检测文本
python main.py detect "基于上述实验结果，我们可以得出以下结论..."

# 改写文本
python main.py rewrite "根据实验数据，该方法在准确率和效率方面均表现优异"

# 指定检查点目录
python main.py detect --checkpoint-dir checkpoints/joint_iter0 "文本"
python main.py rewrite --checkpoint-dir checkpoints/joint_iter0 "文本"
```

## 📊 数据格式

### 检测任务

JSON 列表格式，每个元素包含：
```json
{
  "text": "论文段落文本",
  "label": 0  // 0=人类写作, 1=AI生成
}
```

示例：
```json
[
  {"text": "实验结果表明该方法有效", "label": 1},
  {"text": "我今天去了超市买牛奶", "label": 0}
]
```

### 改写任务

JSON 列表格式，每个元素包含：
```json
{
  "input": "AI生成的文本",
  "output": "改写后的自然表达"
}
```

示例：
```json
[
  {
    "input": "According to the experimental data, we can conclude...",
    "output": "实验结果表明..."
  }
]
```

## 🔧 技术细节

### 双 LoRA 架构

- **基础模型**: Qwen3.5-2B (4bit 量化)
- **检测 LoRA**:
  - Task Type: `SEQ_CLS`
  - Rank: `r=8`
  - Alpha: `16`
  - 额外分类头: `Linear(hidden_size, 2)`
- **改写 LoRA**:
  - Task Type: `CAUSAL_LM`
  - Rank: `r=16`
  - Alpha: `32`

### 交替训练策略

```
for iteration in range(N):
    train_detection(epochs_per_iter)  # 训练检测任务
    train_rewriting(epochs_per_iter)  # 训练改写任务
```

优点：
- 两个任务都能持续学习
- 避免任务间的梯度冲突
- 显存占用稳定（一次只训练一个 adapter）

### 评估指标

**检测任务**:
- Accuracy, Precision, Recall, F1
- AUC-ROC
- 混淆矩阵
- 特异度 (Specificity)

**改写任务**:
- BLEU
- ROUGE (1, 2, L)
- BERTScore (P, R, F1)
- 语义相似度 (Sentence Transformer)

## 📝 代码示例

### Python API 使用

```python
from infer.predictor import YetyPredictor

# 初始化预测器
predictor = YetyPredictor(
    checkpoint_dir="checkpoints/joint_iter0",
    base_model_name="Qwen/Qwen3.5-2B"
)

# 检测
result = predictor.detect("论文文本")
print(f"标签: {result['label']}, 置信度: {result['confidence']:.2%}")

# 改写
result = predictor.rewrite("AI生成的文本")
print(f"改写结果: {result['rewritten_text']}")
```

### 自定义训练

```python
from models.multitask_model import MultiTaskModel
from train.train_alternating import train_alternating

# 初始化模型
model_manager = MultiTaskModel(
    base_model_name="Qwen/Qwen3.5-2B",
    model_config=model_config
)

# 自定义数据路径
detection_data = {"train": "...", "eval": "..."}
rewriting_data = {"train": "...", "eval": "..."}

# 训练
history = train_alternating(
    model_manager,
    detection_data,
    rewriting_data,
    training_config,
    model_config,
    iterations=5
)
```

## 📊 预期性能

基于 4GB 显存 (RTX 3050) 和示例数据：

| 任务 | 预期效果 | 所需时间（2轮） |
|------|---------|---------------|
| 检测 | F1 ~ 0.85-0.90 | 2-4 小时 |
| 改写 | BLEU ~ 30-40 | 4-6 小时 |

实际性能取决于：
- 数据质量和数量
- 训练轮数
- 硬件配置

## 🐛 常见问题

**Q: 显存不足怎么办？**
A: 可以尝试：
   - 减小 batch_size（训练配置）
   - 降低 LoRA rank（r=4 或 r=2）
   - 使用更小的基础模型（如 Qwen1.5-1.8B）

**Q: 两个任务效果都不好？**
A: 检查：
   - 数据是否足够（检测至少 3k，改写至少 5k）
   - 数据质量（标签是否正确，改写是否合理）
   - 学习率是否合适（可以尝试 1e-4 或 5e-4）

**Q: 如何部署到生产环境？**
A:
   - 保存最终的 adapter
   - 使用 `YetyPredictor` 封装 API
   - 考虑使用 FastAPI/Flask 暴露 HTTP 接口
   - 使用 vLLM/Ollama 加速推理

## 📚 参考

- [PEFT 文档](https://huggingface.co/docs/peft)
- [Qwen 模型](https://huggingface.co/Qwen)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)

## 📄 License

MIT License

---

**作者**: 宜小听 (Yi Xiaoting)
**版本**: 0.1.0
**日期**: 2026-03-28
