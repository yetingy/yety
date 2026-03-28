# Google Colab 训练指南

## 1. 打开 Colab
访问 https://colab.research.google.com/

## 2. 连接 GPU
- 点击 `修改 → 笔记本设置`
- 硬件加速器选择 `T4 GPU`
- 保存

## 3. 运行以下命令

```python
# 克隆项目
!git clone https://github.com/你的用户名/yety.git
%cd yety

# 安装依赖
!pip install transformers peft bitsandbytes accelerate datasets scikit-learn

# 下载数据（如果需要）
# !python scripts/download_data.py

# 开始训练
!python main.py train --iterations 3 --epochs 2
```

## 4. 训练完成后下载模型

训练好的模型在 `checkpoints/` 目录下，可以使用以下命令下载：

```python
# 列出 checkpoints 目录
import os
print(os.listdir('checkpoints/'))

# 下载整个文件夹
from google.colab import files
import shutil
shutil.make_archive('checkpoints', 'zip', 'checkpoints')
files.download('checkpoints.zip')
```

## 注意事项

- 免费版 Colab 每天最多使用 12 小时
- 建议先在小数据集上测试
- T4 GPU 显存约 15GB，足够训练 Qwen3.5-2B
