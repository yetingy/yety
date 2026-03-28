"""
自定义 Trainer - 支持检测任务的分类头训练
"""

import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class DetectionTrainer(Trainer):
    """
    检测任务的自定义 Trainer
    
    需要处理分类头的训练，计算分类 loss
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算检测任务的 loss

        Args:
            model: 模型（包含 classifier 头）
            inputs: 输入字典，包含 input_ids, attention_mask, labels
            return_outputs: 是否返回 outputs
            num_items_in_batch: batch 中的样本数量（新版 transformers 需要）

        Returns:
            loss 或 (loss, outputs)
        """
        # 前向传播
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
            output_hidden_states=True  # 请求返回隐藏状态
        )

        # Qwen 模型：hidden_states 是一个元组 (layer1_hidden, layer2_hidden, ..., last_layer_hidden)
        # 形状是 [batch, seq_len, hidden_size]
        if outputs.hidden_states is not None:
            # 取最后一层的隐藏状态
            last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
        else:
            # Fallback: 使用 logits 但需要注意形状变换
            # logits 形状是 [batch, seq_len, vocab_size]
            last_hidden = outputs.logits

        # 取序列第一个 token 的隐藏状态（相当于 [CLS] 位置）
        cls_token = last_hidden[:, 0, :]  # [batch, hidden_size]

        # 确保 cls_token 与分类头 dtype 一致
        classifier_dtype = model.classifier.weight.dtype
        cls_token = cls_token.to(dtype=classifier_dtype)

        # 通过分类头
        logits = model.classifier(cls_token)  # [batch, num_labels]

        # 计算交叉熵 loss
        loss = torch.nn.functional.cross_entropy(
            logits,
            inputs["labels"]
        )

        if return_outputs:
            return loss, {"logits": logits, "outputs": outputs}
        return loss
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        预测步骤 - 返回 logits 用于评估
        """
        with torch.no_grad():
            loss, outputs_dict = self.compute_loss(model, inputs, return_outputs=True, num_items_in_batch=None)
            logits = outputs_dict["logits"]

        if prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        return (loss, logits, labels)


class RewritingTrainer(Trainer):
    """
    改写任务的自定义 Trainer
    
    使用标准的语言模型 loss，但需要处理 label masking
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算改写任务的 loss

        注意：inputs 中的 labels 已经将输入部分设为 -100
        """
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            return_dict=True
        )
        
        loss = outputs.loss
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        预测步骤 - 返回生成的文本用于评估
        """
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # 对于生成任务，评估时需要生成文本而不是返回 logits
        # 这里简化，返回 None，评估时单独处理
        return (loss, None, inputs["labels"])


def compute_metrics_detection(eval_pred):
    """
    计算检测任务的评估指标
    
    Args:
        eval_pred: EvalPrediction 对象，包含 predictions 和 label_ids
        
    Returns:
        指标字典
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_metrics_rewriting(eval_pred, tokenizer):
    """
    计算改写任务的评估指标（BLEU, ROUGE 等）
    
    注意：这里需要生成文本，所以不能直接用 logits
    实际评估应该在预测阶段单独进行
    """
    # TODO: 实现 BLEU, ROUGE 等指标
    return {}
