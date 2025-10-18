"""
正确的分层排序评估器

使用真实表达值进行评估，修复所有评估bug
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)


class CorrectStratifiedEvaluator:
    """正确的分层排序评估器（使用真实表达值）"""
    
    def __init__(self):
        pass
    
    def _collect_group_data(self, model, data_loader, device) -> Dict[str, Dict]:
        """收集每个组的完整数据"""
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        
        model.eval()
        
        # 按组收集：group_key -> {seq_idx -> {pred_score, true_expr}}
        group_data = defaultdict(lambda: {
            'sequences': {},  # seq_idx -> {pred_score, true_expr, count}
            'pairs': [],  # (pred_i, pred_j, expr_i, expr_j, label)
            'group_size': 0
        })
        
        total_loss = 0.0
        total_pairs = 0
        
        with torch.no_grad():
            for batch in data_loader:
                features_i = batch['features_i'].to(device)
                features_j = batch['features_j'].to(device)
                labels = batch['label'].to(device)
                group_keys = batch['group_key']
                expr_i = batch['expr_i']  # numpy array
                expr_j = batch['expr_j']
                seq_idx_i = batch['seq_idx_i']
                seq_idx_j = batch['seq_idx_j']
                group_sizes = batch['group_size']
                
                # 前向传播
                scores_i = model(features_i)
                scores_j = model(features_j)
                
                # 计算损失
                from codon_verifier.improved_ranking_model import ImprovedMarginRankingLoss
                criterion = ImprovedMarginRankingLoss()
                loss = criterion(scores_i, scores_j, labels)
                total_loss += loss.item()
                total_pairs += labels.size(0)
                
                # 收集数据
                for idx in range(len(group_keys)):
                    gk = group_keys[idx]
                    
                    # 记录序列级别的数据
                    idx_i = seq_idx_i[idx].item()
                    idx_j = seq_idx_j[idx].item()
                    score_i = scores_i[idx].item()
                    score_j = scores_j[idx].item()
                    true_i = expr_i[idx].item() if hasattr(expr_i[idx], 'item') else float(expr_i[idx])
                    true_j = expr_j[idx].item() if hasattr(expr_j[idx], 'item') else float(expr_j[idx])
                    
                    # 累积预测得分（处理同一序列在多个pairs中出现）
                    if idx_i not in group_data[gk]['sequences']:
                        group_data[gk]['sequences'][idx_i] = {
                            'pred_scores': [], 'true_expr': true_i
                        }
                    group_data[gk]['sequences'][idx_i]['pred_scores'].append(score_i)
                    
                    if idx_j not in group_data[gk]['sequences']:
                        group_data[gk]['sequences'][idx_j] = {
                            'pred_scores': [], 'true_expr': true_j
                        }
                    group_data[gk]['sequences'][idx_j]['pred_scores'].append(score_j)
                    
                    # 记录pair
                    group_data[gk]['pairs'].append((
                        score_i, score_j, true_i, true_j, labels[idx].item()
                    ))
                    group_data[gk]['group_size'] = group_sizes[idx].item()
        
        # 对每个序列的预测得分取平均
        for gk in group_data:
            for seq_idx in group_data[gk]['sequences']:
                scores = group_data[gk]['sequences'][seq_idx]['pred_scores']
                group_data[gk]['sequences'][seq_idx]['pred_score'] = np.mean(scores)
        
        return dict(group_data), total_loss / total_pairs if total_pairs > 0 else 0.0
    
    def _compute_group_metrics(self, group_data: Dict) -> Dict[str, float]:
        """计算单个组的指标"""
        sequences = group_data['sequences']
        pairs = group_data['pairs']
        
        if len(sequences) < 2:
            return None
        
        # 提取预测得分和真实表达值
        seq_indices = sorted(sequences.keys())
        pred_scores = [sequences[idx]['pred_score'] for idx in seq_indices]
        true_exprs = [sequences[idx]['true_expr'] for idx in seq_indices]
        
        metrics = {}
        
        # 1. Spearman相关系数（预测排名 vs 真实排名）
        if len(set(true_exprs)) > 1:  # 确保真实值有变化
            try:
                spearman, _ = spearmanr(pred_scores, true_exprs)
                metrics['spearman'] = spearman if not np.isnan(spearman) else None
            except:
                metrics['spearman'] = None
        else:
            metrics['spearman'] = None
        
        # 2. Top-1准确率
        pred_top1_idx = seq_indices[np.argmax(pred_scores)]
        true_top1_idx = seq_indices[np.argmax(true_exprs)]
        metrics['top1'] = 1.0 if pred_top1_idx == true_top1_idx else 0.0
        
        # 3. Top-3准确率
        if len(seq_indices) >= 3:
            pred_top3_indices = set([seq_indices[i] for i in np.argsort(pred_scores)[-3:]])
            true_top3_indices = set([seq_indices[i] for i in np.argsort(true_exprs)[-3:]])
            metrics['top3'] = len(pred_top3_indices & true_top3_indices) / 3.0
        else:
            metrics['top3'] = None
        
        # 4. Pairwise准确率（严格版本：平手按0.5）
        if len(pairs) > 0:
            correct = 0.0
            for score_i, score_j, expr_i, expr_j, label in pairs:
                # 真实排序
                true_order = 1 if expr_i > expr_j else (0 if expr_i < expr_j else 0.5)
                # 预测排序
                pred_order = 1 if score_i > score_j else (0 if score_i < score_j else 0.5)
                
                # 如果真实是平手，跳过或按0.5
                if abs(expr_i - expr_j) < 1e-6:
                    correct += 0.5
                else:
                    # 比较预测和真实
                    if (true_order == 1 and pred_order == 1) or (true_order == 0 and pred_order == 0):
                        correct += 1.0
                    elif pred_order == 0.5:  # 预测平手
                        correct += 0.5
            
            metrics['pairwise_acc'] = correct / len(pairs)
        else:
            metrics['pairwise_acc'] = None
        
        return metrics
    
    def _stratify_groups(self, all_group_data: Dict) -> Dict[str, Dict]:
        """按组大小分层"""
        stratified = {
            'trivial': {},   # n=1
            'small': {},     # n=2-3
            'medium': {},    # n=4-6
            'large': {}      # n>=7
        }
        
        for gk, gdata in all_group_data.items():
            n = len(gdata['sequences'])
            
            if n == 1:
                stratified['trivial'][gk] = gdata
            elif 2 <= n <= 3:
                stratified['small'][gk] = gdata
            elif 4 <= n <= 6:
                stratified['medium'][gk] = gdata
            else:
                stratified['large'][gk] = gdata
        
        return stratified
    
    def _evaluate_stratum(self, stratum_groups: Dict, stratum_name: str) -> Dict[str, Any]:
        """评估单个层级"""
        if not stratum_groups:
            return {
                'n_groups': 0,
                'spearman': None,
                'top1': None,
                'top3': None,
                'pairwise_acc': None,
                'random_top1_baseline': None
            }
        
        # 计算每组的指标
        group_metrics_list = []
        for gk, gdata in stratum_groups.items():
            metrics = self._compute_group_metrics(gdata)
            if metrics:
                group_metrics_list.append(metrics)
        
        if not group_metrics_list:
            return {
                'n_groups': len(stratum_groups),
                'spearman': None,
                'top1': None,
                'top3': None,
                'pairwise_acc': None,
                'random_top1_baseline': None
            }
        
        # 宏平均
        def safe_mean(values):
            valid = [v for v in values if v is not None]
            return np.mean(valid) if valid else None
        
        def safe_median(values):
            valid = [v for v in values if v is not None]
            return np.median(valid) if valid else None
        
        def safe_std(values):
            valid = [v for v in values if v is not None]
            return np.std(valid) if valid else None
        
        def safe_ci(values, confidence=0.95):
            valid = [v for v in values if v is not None]
            if len(valid) < 2:
                return None, None
            alpha = 1 - confidence
            lower = np.percentile(valid, alpha/2 * 100)
            upper = np.percentile(valid, (1 - alpha/2) * 100)
            return lower, upper
        
        spearman_values = [m['spearman'] for m in group_metrics_list]
        top1_values = [m['top1'] for m in group_metrics_list]
        top3_values = [m['top3'] for m in group_metrics_list]
        pairwise_values = [m['pairwise_acc'] for m in group_metrics_list]
        
        # 计算随机基线
        group_sizes = [len(gdata['sequences']) for gdata in stratum_groups.values()]
        random_top1 = np.mean([1.0/size for size in group_sizes])
        random_top3 = np.mean([min(3.0/size, 1.0) for size in group_sizes])
        
        spearman_ci = safe_ci(spearman_values)
        top1_ci = safe_ci(top1_values)
        
        results = {
            'n_groups': len(stratum_groups),
            'avg_group_size': np.mean(group_sizes),
            # Spearman
            'spearman': safe_mean(spearman_values),
            'spearman_median': safe_median(spearman_values),
            'spearman_std': safe_std(spearman_values),
            'spearman_ci': spearman_ci,
            # Top-1
            'top1': safe_mean(top1_values),
            'top1_median': safe_median(top1_values),
            'top1_std': safe_std(top1_values),
            'top1_ci': top1_ci,
            'random_top1_baseline': random_top1,
            # Top-3
            'top3': safe_mean(top3_values),
            # Pairwise Accuracy
            'pairwise_acc': safe_mean(pairwise_values),
            'pairwise_median': safe_median(pairwise_values),
            'pairwise_std': safe_std(pairwise_values),
            # Random baselines
            'random_top3_baseline': random_top3
        }
        
        # Log
        logger.info(f"\n  [{stratum_name}] n_groups={results['n_groups']}, avg_size={results['avg_group_size']:.1f}")
        if results['spearman'] is not None:
            logger.info(f"    Spearman: {results['spearman']:.4f} (median={results['spearman_median']:.4f}, std={results['spearman_std']:.4f})")
            if results['spearman_ci'][0] is not None:
                logger.info(f"              95% CI: [{results['spearman_ci'][0]:.4f}, {results['spearman_ci'][1]:.4f}]")
        if results['top1'] is not None:
            logger.info(f"    Top-1: {results['top1']:.4f} (median={results['top1_median']:.4f})")
            logger.info(f"           vs Random={results['random_top1_baseline']:.4f}, Improvement={results['top1']-results['random_top1_baseline']:.4f}")
            if results['top1_ci'][0] is not None:
                logger.info(f"           95% CI: [{results['top1_ci'][0]:.4f}, {results['top1_ci'][1]:.4f}]")
        if results['pairwise_acc'] is not None:
            logger.info(f"    Pairwise Acc: {results['pairwise_acc']:.4f} (median={results['pairwise_median']:.4f})")
        
        return results
    
    def evaluate_stratified(self, model, data_loader, device) -> Dict[str, Any]:
        """分层评估模型性能"""
        logger.info("\n" + "="*80)
        logger.info("Stratified Evaluation (Using True Expression Values)")
        logger.info("="*80)
        
        # 收集数据
        all_group_data, avg_loss = self._collect_group_data(model, data_loader, device)
        
        # 分层
        stratified_groups = self._stratify_groups(all_group_data)
        
        results = {
            'avg_loss': avg_loss,
            'total_groups': len(all_group_data)
        }
        
        # 评估每个层级
        for stratum_name in ['trivial', 'small', 'medium', 'large']:
            stratum_groups = stratified_groups[stratum_name]
            stratum_results = self._evaluate_stratum(stratum_groups, stratum_name)
            results[stratum_name] = stratum_results
        
        # 重点：medium+large组合
        medium_large_groups = {**stratified_groups['medium'], **stratified_groups['large']}
        if medium_large_groups:
            logger.info(f"\n⭐ Key Metrics (Medium + Large groups):")
            medium_large_results = self._evaluate_stratum(medium_large_groups, 'medium+large')
            results['medium_large_combined'] = medium_large_results
        
        # 汇总
        logger.info("\n" + "="*80)
        logger.info("Summary:")
        logger.info(f"  Total groups: {results['total_groups']}")
        logger.info(f"  Avg loss: {results['avg_loss']:.4f}")
        logger.info(f"  Distribution: trivial={results['trivial']['n_groups']}, " +
                   f"small={results['small']['n_groups']}, " +
                   f"medium={results['medium']['n_groups']}, " +
                   f"large={results['large']['n_groups']}")
        logger.info("="*80)
        
        return results

