"""
修复的排序模型评估器

实现按组宏平均评估，而不是全局混合评估
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)

class FixedRankingEvaluator:
    """修复的排序模型评估器"""
    
    def __init__(self):
        self.group_metrics = defaultdict(list)
    
    def evaluate_by_groups(self, model, data_loader, device) -> Dict[str, Any]:
        # 处理设备配置
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        """按组评估模型性能"""
        model.eval()
        
        # 按组收集预测和标签
        group_predictions = defaultdict(list)
        group_labels = defaultdict(list)
        group_pairs = defaultdict(list)
        
        total_loss = 0.0
        total_pairs = 0
        
        with torch.no_grad():
            for batch in data_loader:
                features_i = batch['features_i'].to(device)
                features_j = batch['features_j'].to(device)
                labels = batch['label'].to(device)
                group_keys = batch['group_key']
                
                # 前向传播
                scores_i = model(features_i)
                scores_j = model(features_j)
                
                # 计算损失
                from codon_verifier.improved_ranking_model import ImprovedMarginRankingLoss
                criterion = ImprovedMarginRankingLoss()
                loss = criterion(scores_i, scores_j, labels)
                total_loss += loss.item()
                total_pairs += labels.size(0)
                
                # 按组收集数据
                for i in range(len(group_keys)):
                    group_key = group_keys[i]
                    group_predictions[group_key].extend([scores_i[i].item(), scores_j[i].item()])
                    group_labels[group_key].extend([1, 0] if labels[i] == 1 else [0, 1])
                    group_pairs[group_key].append((scores_i[i].item(), scores_j[i].item(), labels[i].item()))
        
        # 计算按组的指标
        group_spearman_scores = []
        group_accuracies = []
        group_top1_scores = []
        group_top3_scores = []
        
        for group_key in group_predictions:
            pairs = group_pairs[group_key]
            if len(pairs) < 2:  # 至少需要2个pair才能计算Spearman
                continue
            
            # 计算组内Spearman相关系数
            predictions = [pair[0] for pair in pairs] + [pair[1] for pair in pairs]
            labels = [pair[2] for pair in pairs] + [1 - pair[2] for pair in pairs]
            
            if len(set(labels)) > 1:  # 确保标签有变化
                spearman_corr, _ = spearmanr(predictions, labels)
                if not np.isnan(spearman_corr):
                    group_spearman_scores.append(spearman_corr)
            
            # 计算组内排序准确率
            correct = 0
            total = len(pairs)
            for pair in pairs:
                score_i, score_j, label = pair
                if (score_i > score_j and label == 1) or (score_i < score_j and label == 0):
                    correct += 1
            
            if total > 0:
                group_accuracies.append(correct / total)
            
            # 计算Top-k准确率（这里简化为Top-1）
            if len(pairs) >= 1:
                # 对于Top-1，我们检查第一个pair是否正确
                first_pair = pairs[0]
                score_i, score_j, label = first_pair
                if (score_i > score_j and label == 1) or (score_i < score_j and label == 0):
                    group_top1_scores.append(1.0)
                else:
                    group_top1_scores.append(0.0)
        
        # 计算宏平均指标
        macro_spearman = np.mean(group_spearman_scores) if group_spearman_scores else 0.0
        macro_accuracy = np.mean(group_accuracies) if group_accuracies else 0.0
        macro_top1 = np.mean(group_top1_scores) if group_top1_scores else 0.0
        
        # 计算中位数和分位数
        spearman_median = np.median(group_spearman_scores) if group_spearman_scores else 0.0
        spearman_p25 = np.percentile(group_spearman_scores, 25) if group_spearman_scores else 0.0
        spearman_p75 = np.percentile(group_spearman_scores, 75) if group_spearman_scores else 0.0
        
        accuracy_median = np.median(group_accuracies) if group_accuracies else 0.0
        accuracy_p25 = np.percentile(group_accuracies, 25) if group_accuracies else 0.0
        accuracy_p75 = np.percentile(group_accuracies, 75) if group_accuracies else 0.0
        
        results = {
            'macro_spearman': macro_spearman,
            'macro_accuracy': macro_accuracy,
            'macro_top1': macro_top1,
            'spearman_median': spearman_median,
            'spearman_p25': spearman_p25,
            'spearman_p75': spearman_p75,
            'accuracy_median': accuracy_median,
            'accuracy_p25': accuracy_p25,
            'accuracy_p75': accuracy_p75,
            'n_groups': len(group_predictions),
            'n_groups_with_metrics': len(group_spearman_scores),
            'avg_loss': total_loss / total_pairs if total_pairs > 0 else 0.0,
            'group_spearman_scores': group_spearman_scores,
            'group_accuracies': group_accuracies
        }
        
        logger.info(f"Group-wise evaluation results:")
        logger.info(f"  Groups evaluated: {len(group_predictions)}")
        logger.info(f"  Groups with metrics: {len(group_spearman_scores)}")
        logger.info(f"  Macro Spearman: {macro_spearman:.4f} (median: {spearman_median:.4f})")
        logger.info(f"  Macro Accuracy: {macro_accuracy:.4f} (median: {accuracy_median:.4f})")
        logger.info(f"  Macro Top-1: {macro_top1:.4f}")
        
        return results
    
    def bootstrap_confidence_interval(self, scores: List[float], n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
        """计算bootstrap置信区间"""
        if len(scores) < 2:
            return 0.0, 0.0
        
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return ci_lower, ci_upper
    
    def evaluate_with_confidence_intervals(self, model, data_loader, device) -> Dict[str, Any]:
        """带置信区间的评估"""
        results = self.evaluate_by_groups(model, data_loader, device)
        
        # 计算bootstrap置信区间
        if results['group_spearman_scores']:
            spearman_ci = self.bootstrap_confidence_interval(results['group_spearman_scores'])
            results['spearman_ci_lower'] = spearman_ci[0]
            results['spearman_ci_upper'] = spearman_ci[1]
        else:
            results['spearman_ci_lower'] = 0.0
            results['spearman_ci_upper'] = 0.0
        
        if results['group_accuracies']:
            accuracy_ci = self.bootstrap_confidence_interval(results['group_accuracies'])
            results['accuracy_ci_lower'] = accuracy_ci[0]
            results['accuracy_ci_upper'] = accuracy_ci[1]
        else:
            results['accuracy_ci_lower'] = 0.0
            results['accuracy_ci_upper'] = 0.0
        
        logger.info(f"Confidence intervals (95%):")
        logger.info(f"  Spearman: [{results['spearman_ci_lower']:.4f}, {results['spearman_ci_upper']:.4f}]")
        logger.info(f"  Accuracy: [{results['accuracy_ci_lower']:.4f}, {results['accuracy_ci_upper']:.4f}]")
        
        return results
    
    def evaluate_per_host(self, model, data_loader, device) -> Dict[str, Dict[str, Any]]:
        # 处理设备配置
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        """按宿主评估模型性能"""
        model.eval()
        
        # 按宿主分组收集数据
        host_groups = defaultdict(list)
        
        with torch.no_grad():
            for batch in data_loader:
                features_i = batch['features_i'].to(device)
                features_j = batch['features_j'].to(device)
                labels = batch['label'].to(device)
                group_keys = batch['group_key']
                
                # 前向传播
                scores_i = model(features_i)
                scores_j = model(features_j)
                
                # 按宿主分组
                for i in range(len(group_keys)):
                    group_key = group_keys[i]
                    # 从group_key中提取宿主信息
                    if '_' in group_key:
                        host = group_key.split('_')[-1]
                        host_groups[host].append({
                            'score_i': scores_i[i].item(),
                            'score_j': scores_j[i].item(),
                            'label': labels[i].item()
                        })
        
        # 计算每个宿主的指标
        host_results = {}
        for host, pairs in host_groups.items():
            if len(pairs) < 2:
                continue
            
            # 计算该宿主的指标
            predictions = [p['score_i'] for p in pairs] + [p['score_j'] for p in pairs]
            labels = [p['label'] for p in pairs] + [1 - p['label'] for p in pairs]
            
            if len(set(labels)) > 1:
                spearman_corr, _ = spearmanr(predictions, labels)
                if not np.isnan(spearman_corr):
                    host_results[host] = {
                        'spearman': spearman_corr,
                        'n_pairs': len(pairs),
                        'accuracy': sum(1 for p in pairs if (p['score_i'] > p['score_j'] and p['label'] == 1) or (p['score_i'] < p['score_j'] and p['label'] == 0)) / len(pairs)
                    }
        
        logger.info(f"Per-host evaluation results:")
        for host, metrics in host_results.items():
            logger.info(f"  {host}: Spearman={metrics['spearman']:.4f}, Accuracy={metrics['accuracy']:.4f}, Pairs={metrics['n_pairs']}")
        
        return host_results
