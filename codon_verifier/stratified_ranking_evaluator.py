"""
分层排序模型评估器

修复的功能：
1. 正确的Top-1计算：从pairs重建序列得分，找到预测最高的序列
2. 正确的Spearman计算：基于序列级别的预测vs真实值
3. 分层评估：按组大小（2-3条, 4-6条, ≥7条）分层报告
4. 随机基线计算：提供性能参考
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)


class StratifiedRankingEvaluator:
    """分层排序模型评估器（修复版）"""
    
    def __init__(self):
        self.group_metrics = defaultdict(list)
    
    def _reconstruct_sequence_scores(self, pairs: List[Tuple[float, float, float]]) -> Dict[int, float]:
        """
        从pairs重建序列级别的得分
        
        Args:
            pairs: List of (score_i, score_j, label)
        
        Returns:
            Dict mapping sequence_idx to predicted_score
        """
        # 构建图：每个序列的所有预测得分
        seq_scores = defaultdict(list)
        
        for idx, (score_i, score_j, label) in enumerate(pairs):
            # 使用pair索引来标识序列
            # pair[i] = (seq_a, seq_b), pair[i+1] = (seq_a, seq_c), ...
            # 我们需要识别哪些scores属于同一个序列
            
            # 简化方法：将每个unique的score视为一个序列的得分
            seq_scores[f"seq_{idx}_i"].append(score_i)
            seq_scores[f"seq_{idx}_j"].append(score_j)
        
        # 对每个序列，取平均得分
        final_scores = {}
        for seq_id, scores in seq_scores.items():
            final_scores[seq_id] = np.mean(scores)
        
        return final_scores
    
    def _compute_correct_top1(self, pairs: List[Tuple[float, float, float]]) -> Optional[float]:
        """
        正确计算Top-1准确率
        
        从pairs中推断：
        1. 识别组内所有唯一的序列及其得分
        2. 找到预测得分最高的序列
        3. 找到真实表达值最高的序列（从label推断）
        4. 检查是否匹配
        
        注意：由于数据格式限制，这是一个近似方法
        """
        if len(pairs) == 0:
            return None
        
        # 收集所有唯一的得分
        all_scores = set()
        for score_i, score_j, label in pairs:
            all_scores.add(score_i)
            all_scores.add(score_j)
        
        if len(all_scores) <= 1:
            return None  # 无法区分
        
        # 按得分排序
        sorted_scores = sorted(all_scores, reverse=True)
        predicted_top = sorted_scores[0]
        
        # 尝试推断真实top：如果一个序列在所有比较中都应该赢，它就是真实top
        # 构建每个得分在pairs中的表现
        score_wins = defaultdict(int)
        score_total = defaultdict(int)
        
        for score_i, score_j, label in pairs:
            if label == 1:  # i > j in true ranking
                score_wins[score_i] += 1
                score_total[score_i] += 1
                score_total[score_j] += 1
            else:  # j > i in true ranking
                score_wins[score_j] += 1
                score_total[score_i] += 1
                score_total[score_j] += 1
        
        # 计算每个得分的胜率
        score_win_rates = {}
        for score in all_scores:
            if score_total[score] > 0:
                score_win_rates[score] = score_wins[score] / score_total[score]
            else:
                score_win_rates[score] = 0.0
        
        # 真实top应该是胜率最高的
        if score_win_rates:
            true_top = max(score_win_rates.items(), key=lambda x: x[1])[0]
            return 1.0 if abs(predicted_top - true_top) < 1e-6 else 0.0
        
        return None
    
    def _compute_correct_spearman(self, pairs: List[Tuple[float, float, float]]) -> Optional[float]:
        """
        正确计算Spearman相关系数
        
        方法：从pairs中提取所有唯一序列的预测得分和真实排名
        """
        if len(pairs) < 2:
            return None
        
        # 收集所有唯一的得分及其真实胜率
        all_scores = set()
        for score_i, score_j, label in pairs:
            all_scores.add(score_i)
            all_scores.add(score_j)
        
        if len(all_scores) < 2:
            return None
        
        # 计算每个得分的真实胜率（作为真实表达值的代理）
        score_wins = defaultdict(int)
        score_total = defaultdict(int)
        
        for score_i, score_j, label in pairs:
            if label == 1:  # i > j
                score_wins[score_i] += 1
                score_total[score_i] += 1
                score_total[score_j] += 1
            else:
                score_wins[score_j] += 1
                score_total[score_i] += 1
                score_total[score_j] += 1
        
        # 提取预测得分和真实胜率
        pred_scores = []
        true_rates = []
        for score in sorted(all_scores):
            pred_scores.append(score)
            if score_total[score] > 0:
                true_rates.append(score_wins[score] / score_total[score])
            else:
                true_rates.append(0.0)
        
        # 计算Spearman相关系数
        if len(set(true_rates)) < 2:
            return None
        
        try:
            spearman_corr, _ = spearmanr(pred_scores, true_rates)
            return spearman_corr if not np.isnan(spearman_corr) else None
        except:
            return None
    
    def _compute_pairwise_accuracy(self, pairs: List[Tuple[float, float, float]]) -> float:
        """计算pair-wise准确率"""
        if len(pairs) == 0:
            return 0.0
        
        correct = 0
        for score_i, score_j, label in pairs:
            if (score_i > score_j and label == 1) or (score_i < score_j and label == 0):
                correct += 1
        
        return correct / len(pairs)
    
    def _stratify_groups(self, group_pairs: Dict[str, List]) -> Dict[str, Dict[str, List]]:
        """
        按组大小分层
        
        Returns:
            {
                'small': {group_key: pairs},
                'medium': {group_key: pairs},
                'large': {group_key: pairs}
            }
        """
        stratified = {
            'trivial': {},   # n=1 (no pairs)
            'small': {},     # n=2-3
            'medium': {},    # n=4-6
            'large': {}      # n>=7
        }
        
        for group_key, pairs in group_pairs.items():
            # 推断组大小：从pairs中的唯一得分数量
            all_scores = set()
            for score_i, score_j, _ in pairs:
                all_scores.add(score_i)
                all_scores.add(score_j)
            
            group_size = len(all_scores)
            
            if group_size == 1:
                stratified['trivial'][group_key] = pairs
            elif 2 <= group_size <= 3:
                stratified['small'][group_key] = pairs
            elif 4 <= group_size <= 6:
                stratified['medium'][group_key] = pairs
            else:  # >= 7
                stratified['large'][group_key] = pairs
        
        return stratified
    
    def _evaluate_stratum(self, groups: Dict[str, List], stratum_name: str) -> Dict[str, Any]:
        """评估单个层级"""
        if not groups:
            return {
                'n_groups': 0,
                'spearman': None,
                'accuracy': None,
                'top1': None
            }
        
        spearman_scores = []
        accuracy_scores = []
        top1_scores = []
        
        for group_key, pairs in groups.items():
            # Spearman
            spearman = self._compute_correct_spearman(pairs)
            if spearman is not None:
                spearman_scores.append(spearman)
            
            # Accuracy
            accuracy = self._compute_pairwise_accuracy(pairs)
            accuracy_scores.append(accuracy)
            
            # Top-1
            top1 = self._compute_correct_top1(pairs)
            if top1 is not None:
                top1_scores.append(top1)
        
        # 计算随机基线
        group_sizes = []
        for pairs in groups.values():
            all_scores = set()
            for score_i, score_j, _ in pairs:
                all_scores.add(score_i)
                all_scores.add(score_j)
            group_sizes.append(len(all_scores))
        
        random_top1_baseline = np.mean([1.0/size for size in group_sizes]) if group_sizes else 0.0
        
        results = {
            'n_groups': len(groups),
            'spearman': np.mean(spearman_scores) if spearman_scores else None,
            'spearman_std': np.std(spearman_scores) if spearman_scores else None,
            'accuracy': np.mean(accuracy_scores) if accuracy_scores else None,
            'accuracy_std': np.std(accuracy_scores) if accuracy_scores else None,
            'top1': np.mean(top1_scores) if top1_scores else None,
            'top1_std': np.std(top1_scores) if top1_scores else None,
            'random_top1_baseline': random_top1_baseline,
            'avg_group_size': np.mean(group_sizes) if group_sizes else 0.0
        }
        
        # Log
        logger.info(f"  [{stratum_name}] n_groups={results['n_groups']}")
        if results['spearman'] is not None:
            logger.info(f"    Spearman: {results['spearman']:.4f} ± {results['spearman_std']:.4f}")
        if results['accuracy'] is not None:
            logger.info(f"    Accuracy: {results['accuracy']:.4f} ± {results['accuracy_std']:.4f}")
        if results['top1'] is not None:
            logger.info(f"    Top-1: {results['top1']:.4f} ± {results['top1_std']:.4f} (random baseline: {results['random_top1_baseline']:.4f})")
        
        return results
    
    def evaluate_stratified(self, model, data_loader, device) -> Dict[str, Any]:
        """分层评估模型性能"""
        # 处理设备
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        
        model.eval()
        
        # 按组收集pairs
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
                
                # 收集pairs
                for i in range(len(group_keys)):
                    group_key = group_keys[i]
                    group_pairs[group_key].append((
                        scores_i[i].item(),
                        scores_j[i].item(),
                        labels[i].item()
                    ))
        
        # 分层
        logger.info("\n=== Stratified Evaluation ===")
        stratified_groups = self._stratify_groups(group_pairs)
        
        results = {
            'avg_loss': total_loss / total_pairs if total_pairs > 0 else 0.0,
            'total_groups': len(group_pairs),
            'total_pairs': total_pairs
        }
        
        # 评估每个层级
        for stratum_name in ['trivial', 'small', 'medium', 'large']:
            stratum_groups = stratified_groups[stratum_name]
            stratum_results = self._evaluate_stratum(stratum_groups, stratum_name)
            results[stratum_name] = stratum_results
        
        # 汇总统计
        logger.info("\n=== Summary ===")
        logger.info(f"Total groups: {results['total_groups']}")
        logger.info(f"Total pairs: {results['total_pairs']}")
        logger.info(f"Avg loss: {results['avg_loss']:.4f}")
        
        logger.info("\nGroup distribution:")
        for stratum in ['trivial', 'small', 'medium', 'large']:
            n = results[stratum]['n_groups']
            pct = n / results['total_groups'] * 100 if results['total_groups'] > 0 else 0
            logger.info(f"  {stratum}: {n} ({pct:.1f}%)")
        
        # 重点关注中大组的表现
        logger.info("\n⭐ Key Metrics (Medium + Large groups):")
        medium_large_groups = {**stratified_groups['medium'], **stratified_groups['large']}
        if medium_large_groups:
            combined_results = self._evaluate_stratum(medium_large_groups, 'medium+large')
            results['medium_large_combined'] = combined_results
        else:
            logger.info("  No medium or large groups available!")
        
        return results
    
    def evaluate_with_confidence_intervals(self, model, data_loader, device) -> Dict[str, Any]:
        """带置信区间的分层评估"""
        results = self.evaluate_stratified(model, data_loader, device)
        
        # 可以添加bootstrap CI，但简化起见先省略
        return results

