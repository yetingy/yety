"""
评估指标计算模块
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, Any, List
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score_score
import torch
from sentence_transformers import SentenceTransformer, util


# 检测任务指标
def compute_detection_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray = None
) -> Dict[str, float]:
    """
    计算检测任务的评估指标
    
    Args:
        predictions: 预测标签 (0 或 1)
        labels: 真实标签
        probabilities: 预测概率 (可选，用于 AUC)
        
    Returns:
        指标字典
    """
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average='binary', zero_division=0),
        "recall": recall_score(labels, predictions, average='binary', zero_division=0),
        "f1": f1_score(labels, predictions, average='binary', zero_division=0),
    }
    
    # 计算 AUC（需要概率）
    if probabilities is not None and len(np.unique(labels)) > 1:
        try:
            auc = roc_auc_score(labels, probabilities[:, 1])
            metrics["auc"] = auc
        except:
            metrics["auc"] = 0.0
    
    # 混淆矩阵
    cm = confusion_matrix(labels, predictions)
    metrics["confusion_matrix"] = cm.tolist()
    
    # 计算特异度 (Specificity = TN / (TN + FP))
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["specificity"] = specificity
    
    return metrics


# 改写任务指标
class RewritingEvaluator:
    """改写任务评估器"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        # 加载语义相似度模型
        print("Loading sentence transformer for semantic similarity...")
        self.sim_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # 初始化 ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
    
    def compute_bleu(
        self,
        hypotheses: List[str],
        references: List[List[str]]
    ) -> Dict[str, float]:
        """
        计算 BLEU 分数
        
        Args:
            hypotheses: 生成的文本列表
            references: 参考文本列表（每个元素是一个参考列表，支持多个参考）
            
        Returns:
            BLEU 分数字典
        """
        bleu = sacrebleu.corpus_bleu(
            hypotheses,
            references,
            tokenize='zh'  # 中文分词
        )
        
        return {
            "bleu": bleu.score,
            "bleu_precision": bleu.precision,
            "bleu_brevity_penalty": bleu.brevity_penalty,
            "bleu_length_ratio": bleu.length_ratio
        }
    
    def compute_rouge(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        计算 ROUGE 分数
        
        Args:
            hypotheses: 生成的文本列表
            references: 参考文本列表
            
        Returns:
            平均 ROUGE 分数
        """
        rouge_scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
        for hyp, ref in zip(hypotheses, references):
            scores = self.rouge_scorer.score(ref, hyp)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)
        
        return {
            "rouge1": np.mean(rouge_scores['rouge1']),
            "rouge2": np.mean(rouge_scores['rouge2']),
            "rougeL": np.mean(rouge_scores['rougeL'])
        }
    
    def compute_bertscore(
        self,
        hypotheses: List[str],
        references: List[str],
        lang: str = "zh"
    ) -> Dict[str, float]:
        """
        计算 BERTScore
        
        Args:
            hypotheses: 生成的文本列表
            references: 参考文本列表
            lang: 语言代码
            
        Returns:
            BERTScore 字典
        """
        P, R, F1 = bert_score_score(
            hypotheses,
            references,
            lang=lang,
            verbose=False
        )
        
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item()
        }
    
    def compute_semantic_similarity(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        计算语义相似度（使用 Sentence Transformer）
        
        Args:
            hypotheses: 生成的文本列表
            references: 参考文本列表
            
        Returns:
            平均相似度分数
        """
        # 编码文本
        hyp_embeddings = self.sim_model.encode(
            hypotheses,
            convert_to_tensor=True,
            device=self.device
        )
        ref_embeddings = self.sim_model.encode(
            references,
            convert_to_tensor=True,
            device=self.device
        )
        
        # 计算余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(hyp_embeddings, ref_embeddings)
        
        return {
            "semantic_similarity": cos_sim.mean().item()
        }
    
    def compute_all_metrics(
        self,
        hypotheses: List[str],
        references: List[str],
        multi_references: List[List[str]] = None
    ) -> Dict[str, float]:
        """
        计算所有改写评估指标
        
        Args:
            hypotheses: 生成的文本列表
            references: 参考文本列表
            multi_references: 多个参考文本列表（可选，用于 BLEU）
            
        Returns:
            所有指标的字典
        """
        metrics = {}
        
        # BLEU
        if multi_references:
            bleu_metrics = self.compute_bleu(hypotheses, multi_references)
        else:
            bleu_metrics = self.compute_bleu(hypotheses, [[ref] for ref in references])
        metrics.update(bleu_metrics)
        
        # ROUGE
        rouge_metrics = self.compute_rouge(hypotheses, references)
        metrics.update(rouge_metrics)
        
        # BERTScore
        bertscore_metrics = self.compute_bertscore(hypotheses, references)
        metrics.update(bertscore_metrics)
        
        # 语义相似度
        sim_metrics = self.compute_semantic_similarity(hypotheses, references)
        metrics.update(sim_metrics)
        
        return metrics


def evaluate_detection(
    model,
    tokenizer,
    test_data: List[Dict[str, Any]],
    max_length: int = 512
) -> Dict[str, float]:
    """
    评估检测模型
    
    Args:
        model: 检测模型（包含 classifier）
        tokenizer: tokenizer
        test_data: 测试数据
        max_length: 最大序列长度
        
    Returns:
        评估指标
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    for item in test_data:
        text = item["text"]
        label = item["label"]
        
        prompt = f"判断以下文本是否由 AI 生成：{text}"
        inputs = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # 获取 [CLS] 位置的 hidden state
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]
                cls_token = hidden_states[:, 0, :]
            else:
                logits = outputs.logits
                cls_token = logits[:, 0, :]
            
            logits = model.classifier(cls_token)
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
        
        all_predictions.append(pred)
        all_labels.append(label)
        all_probabilities.append(probs[0].cpu().numpy())
    
    # 计算指标
    metrics = compute_detection_metrics(
        np.array(all_predictions),
        np.array(all_labels),
        np.array(all_probabilities)
    )
    
    return metrics


def evaluate_rewriting(
    model,
    tokenizer,
    test_data: List[Dict[str, Any]],
    max_length: int = 512,
    max_new_tokens: int = 256,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    评估改写模型
    
    Args:
        model: 改写模型
        tokenizer: tokenizer
        test_data: 测试数据
        max_length: 最大输入长度
        max_new_tokens: 最大生成 token 数
        device: 设备
        
    Returns:
        评估指标
    """
    model.eval()
    
    hypotheses = []
    references = []
    
    for item in test_data:
        input_text = item["input"]
        reference = item["output"]
        
        prompt = f"改写以下 AI 生成的文本，使其更自然：{input_text}\n改写后："
        inputs = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=False,  # 评估时使用 greedy
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated = outputs[0][input_ids.shape[1]:]
        hypothesis = tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        hypotheses.append(hypothesis)
        references.append(reference)
    
    # 计算所有指标
    evaluator = RewritingEvaluator(device=device)
    metrics = evaluator.compute_all_metrics(hypotheses, references)
    
    return metrics


if __name__ == "__main__":
    print("Evaluation metrics module loaded.")
