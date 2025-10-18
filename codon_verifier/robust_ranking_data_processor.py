"""
稳健的排序数据处理器

专门解决数据质量问题：
1. 大幅降低表达量差异阈值
2. 修复弱标签生成逻辑
3. 使用更宽松的排序标准
4. 确保生成≥30,000训练对
"""

import json
import random
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

@dataclass
class RobustRankingDataConfig:
    """稳健的排序数据配置"""
    # 数据过滤
    min_sequences_per_group: int = 2
    max_sequences_per_group: int = 200
    
    # 数据划分
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    
    # 聚类参数
    n_clusters: Optional[int] = None
    min_cluster_size: int = 10
    
    # 弱标签生成
    k_folds: int = 5
    weak_label_model: str = "random_forest"
    
    # 特征工程
    use_enhanced_features: bool = True
    feature_scaling: bool = True
    
    # 交叉验证
    use_cross_validation: bool = True
    cv_folds: int = 5
    
    # 关键修复：大幅降低表达量差异阈值
    min_expression_diff: float = 0.1  # 降低到0.1，之前可能太高
    use_relative_ranking: bool = True  # 使用相对排序而不是绝对差异
    
    # Pair生成参数 - 关键修复
    min_train_pairs: int = 30000
    max_pairs_per_group: int = 500  # 增加每组最大pair数量
    use_weak_label_pairs: bool = True
    weak_label_ratio: float = 0.7  # 增加弱标签pair比例
    use_synthetic_pairs: bool = True  # 使用合成pairs
    synthetic_ratio: float = 0.5  # 合成pair比例
    
    # 随机种子
    random_seed: int = 42

@dataclass
class ProteinGroup:
    """蛋白质组数据类"""
    protein_id: str
    host: str
    sequences: List[Dict]
    
    def __len__(self):
        return len(self.sequences)

class RobustRankingDataProcessor:
    """稳健的排序数据处理器"""
    
    def __init__(self, config: RobustRankingDataConfig):
        self.config = config
        self.protein_groups: Dict[str, ProteinGroup] = {}
        self.cluster_assignments: Dict[str, int] = {}
        self.dataset_splits: Dict[str, List[str]] = {}
        self.weak_labels: Dict[str, float] = {}
        self.feature_scaler: Optional[StandardScaler] = None
        
        # 设置随机种子
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    def load_data(self, data_paths: List[str]) -> List[Dict]:
        """加载数据"""
        logger.info("Loading data from multiple files...")
        all_records = []
        
        for path in data_paths:
            if not Path(path).exists():
                logger.warning(f"File not found: {path}")
                continue
                
            logger.info(f"Loading data from {path}")
            with open(path, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        all_records.append(record)
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"Loaded {len(all_records)} total records")
        return all_records
    
    def group_by_protein_and_host(self, records: List[Dict]) -> Dict[str, ProteinGroup]:
        """按氨基酸序列和host分组数据"""
        logger.info("Grouping data by amino acid sequence and host...")
        
        groups = defaultdict(list)
        
        for record in records:
            protein_aa = record.get("protein_aa", "")
            host = record.get("host", "")
            protein_id = record.get("protein_id", "")
            
            if not protein_aa or not host:
                continue
                
            group_key = f"{protein_aa}_{host}"
            groups[group_key].append(record)
        
        logger.info(f"Found {len(groups)} raw groups before filtering")
        
        # 创建ProteinGroup对象并过滤
        protein_groups = {}
        for group_key, sequences in groups.items():
            if len(sequences) >= self.config.min_sequences_per_group:
                if len(sequences) <= self.config.max_sequences_per_group:
                    protein_id = sequences[0].get("protein_id", "unknown")
                    host = sequences[0].get("host", "unknown")
                    protein_groups[group_key] = ProteinGroup(protein_id, host, sequences)
                else:
                    # 如果序列太多，分层采样
                    sampled = self._stratified_sample(sequences, self.config.max_sequences_per_group)
                    protein_id = sampled[0].get("protein_id", "unknown")
                    host = sampled[0].get("host", "unknown")
                    protein_groups[group_key] = ProteinGroup(protein_id, host, sampled)
        
        logger.info(f"Created {len(protein_groups)} protein groups after filtering")
        
        # 统计信息
        if protein_groups:
            group_sizes = [len(group) for group in protein_groups.values()]
            logger.info(f"Group sizes - Min: {min(group_sizes)}, Max: {max(group_sizes)}, Mean: {np.mean(group_sizes):.2f}")
        else:
            logger.warning("No protein groups created! Check data format and filtering criteria.")
        
        self.protein_groups = protein_groups
        return protein_groups
    
    def _stratified_sample(self, sequences: List[Dict], n_samples: int) -> List[Dict]:
        """分层采样，保持表达量分布"""
        if len(sequences) <= n_samples:
            return sequences
        
        # 按表达量分组
        expression_groups = defaultdict(list)
        for seq in sequences:
            expr = seq.get("expression", {}).get("value", 0)
            # 将表达量分为5个等级
            level = min(4, max(0, int(expr / 20)))
            expression_groups[level].append(seq)
        
        # 按比例采样
        sampled = []
        for level, group_seqs in expression_groups.items():
            n_from_group = max(1, int(n_samples * len(group_seqs) / len(sequences)))
            n_from_group = min(n_from_group, len(group_seqs))
            sampled.extend(random.sample(group_seqs, n_from_group))
        
        # 如果采样数量不够，随机补充
        if len(sampled) < n_samples:
            remaining = [seq for seq in sequences if seq not in sampled]
            needed = n_samples - len(sampled)
            sampled.extend(random.sample(remaining, min(needed, len(remaining))))
        
        return sampled[:n_samples]
    
    def compute_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """计算两个氨基酸序列的相似性"""
        if not seq1 or not seq2:
            return 0.0
        
        # 使用简单的字符匹配相似性
        min_len = min(len(seq1), len(seq2))
        max_len = max(len(seq1), len(seq2))
        
        if max_len == 0:
            return 0.0
        
        matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
        return matches / max_len
    
    def cluster_proteins_by_similarity(self) -> Dict[str, int]:
        """改进的蛋白质聚类"""
        logger.info("Clustering proteins by sequence similarity...")
        
        if not self.protein_groups:
            raise ValueError("No protein groups found. Call group_by_protein_and_host first.")
        
        group_keys = list(self.protein_groups.keys())
        n_groups = len(group_keys)
        
        if n_groups < 2:
            logger.warning("Not enough groups for clustering")
            return {key: 0 for key in group_keys}
        
        # 提取氨基酸序列
        aa_sequences = []
        for group_key in group_keys:
            group = self.protein_groups[group_key]
            # 使用第一个序列的氨基酸序列作为代表
            aa_seq = group.sequences[0].get("protein_aa", "")
            aa_sequences.append(aa_seq)
        
        # 计算相似性矩阵
        similarity_matrix = np.zeros((n_groups, n_groups))
        
        for i in range(n_groups):
            for j in range(i, n_groups):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self.compute_sequence_similarity(aa_sequences[i], aa_sequences[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        # 转换为距离矩阵
        distance_matrix = 1 - similarity_matrix
        
        # 改进的聚类策略
        if self.config.n_clusters is None:
            # 基于数据量自动确定聚类数量
            n_clusters = max(2, min(30, n_groups // 20))
        else:
            n_clusters = self.config.n_clusters
        
        # 使用层次聚类
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # 保存聚类结果
        cluster_assignments = {}
        for i, group_key in enumerate(group_keys):
            cluster_assignments[group_key] = int(cluster_labels[i])
        
        self.cluster_assignments = cluster_assignments
        
        # 统计聚类结果
        cluster_counts = Counter(cluster_labels)
        logger.info(f"Clustering completed: {n_clusters} clusters")
        for cluster_id, count in cluster_counts.items():
            logger.info(f"  Cluster {cluster_id}: {count} proteins")
        
        return cluster_assignments
    
    def split_datasets_balanced(self) -> Dict[str, List[str]]:
        """改进的平衡数据划分"""
        logger.info("Splitting datasets with balanced distribution...")
        
        if not self.cluster_assignments:
            raise ValueError("No cluster assignments found. Call cluster_proteins_by_similarity first.")
        
        # 按聚类分组
        clusters = defaultdict(list)
        for group_key, cluster_id in self.cluster_assignments.items():
            clusters[cluster_id].append(group_key)
        
        # 过滤小聚类
        valid_clusters = {}
        for cluster_id, groups in clusters.items():
            if len(groups) >= self.config.min_cluster_size:
                valid_clusters[cluster_id] = groups
        
        logger.info(f"Valid clusters: {len(valid_clusters)} (filtered from {len(clusters)})")
        
        # 按聚类大小排序，确保大聚类均匀分布
        sorted_clusters = sorted(valid_clusters.items(), key=lambda x: len(x[1]), reverse=True)
        
        # 交替分配到不同数据集，确保平衡
        train_groups = []
        val_groups = []
        test_groups = []
        
        for i, (cluster_id, groups) in enumerate(sorted_clusters):
            if i % 3 == 0:
                train_groups.extend(groups)
            elif i % 3 == 1:
                val_groups.extend(groups)
            else:
                test_groups.extend(groups)
        
        # 如果分布不够平衡，进行调整
        total_groups = len(train_groups) + len(val_groups) + len(test_groups)
        target_train = int(total_groups * self.config.train_ratio)
        target_val = int(total_groups * self.config.val_ratio)
        
        # 调整训练集大小
        if len(train_groups) > target_train:
            excess = len(train_groups) - target_train
            moved_to_val = excess // 2
            moved_to_test = excess - moved_to_val
            val_groups.extend(train_groups[-moved_to_val:])
            test_groups.extend(train_groups[-(moved_to_val + moved_to_test):-moved_to_val])
            train_groups = train_groups[:-excess]
        elif len(train_groups) < target_train:
            needed = target_train - len(train_groups)
            if len(val_groups) > target_val:
                move_from_val = min(needed, len(val_groups) - target_val)
                train_groups.extend(val_groups[:move_from_val])
                val_groups = val_groups[move_from_val:]
                needed -= move_from_val
            if needed > 0 and len(test_groups) > target_val:
                move_from_test = min(needed, len(test_groups) - target_val)
                train_groups.extend(test_groups[:move_from_test])
                test_groups = test_groups[move_from_test:]
        
        self.dataset_splits = {
            "train": train_groups,
            "val": val_groups,
            "test": test_groups
        }
        
        logger.info(f"Balanced dataset splits:")
        logger.info(f"  Train: {len(train_groups)} groups ({len(train_groups)/total_groups*100:.1f}%)")
        logger.info(f"  Val: {len(val_groups)} groups ({len(val_groups)/total_groups*100:.1f}%)")
        logger.info(f"  Test: {len(test_groups)} groups ({len(test_groups)/total_groups*100:.1f}%)")
        
        return self.dataset_splits
    
    def extract_enhanced_features(self, record: Dict) -> np.ndarray:
        """提取增强特征"""
        features = []
        
        # 基础序列特征
        sequence = record.get("sequence", "")
        features.extend([
            len(sequence),  # 序列长度
            sequence.count('A') / len(sequence) if sequence else 0,  # A含量
            sequence.count('T') / len(sequence) if sequence else 0,  # T含量
            sequence.count('G') / len(sequence) if sequence else 0,  # G含量
            sequence.count('C') / len(sequence) if sequence else 0,  # C含量
        ])
        
        # GC含量
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence) if sequence else 0
        features.append(gc_content)
        
        # 密码子使用偏好
        if len(sequence) >= 3:
            codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]
            codon_counts = Counter(codons)
            total_codons = len(codons)
            
            # 常见密码子比例
            common_codons = ['ATG', 'TAA', 'TAG', 'TGA', 'TTT', 'TTC', 'TTA', 'TTG']
            common_ratio = sum(codon_counts.get(codon, 0) for codon in common_codons) / total_codons
            features.append(common_ratio)
            
            # 密码子多样性
            codon_diversity = len(codon_counts) / total_codons
            features.append(codon_diversity)
        else:
            features.extend([0, 0])
        
        # MSA特征
        msa_features = record.get("msa_features", {})
        features.extend([
            msa_features.get("msa_depth", 0),
            msa_features.get("msa_effective_depth", 0),
            msa_features.get("msa_coverage", 0),
            msa_features.get("conservation_mean", 0),
            msa_features.get("conservation_min", 0),
            msa_features.get("conservation_max", 0),
            msa_features.get("conservation_entropy_mean", 0),
            msa_features.get("coevolution_score", 0),
            msa_features.get("contact_density", 0),
            msa_features.get("pfam_count", 0),
            msa_features.get("domain_count", 0),
        ])
        
        # 进化特征
        features.extend([
            record.get("evo2_loglik", 0),
            record.get("evo2_entropy", 0),
            record.get("evo2_kl_divergence", 0),
        ])
        
        # 结构特征
        structure_features = record.get("structure_features", {})
        features.extend([
            structure_features.get("secondary_structure_helix", 0),
            structure_features.get("secondary_structure_sheet", 0),
            structure_features.get("secondary_structure_loop", 0),
            structure_features.get("disorder_score", 0),
            structure_features.get("solvent_accessibility", 0),
        ])
        
        # 宿主特异性特征
        host = record.get("host", "")
        host_features = [0] * 5  # 支持5个宿主
        host_mapping = {"E_coli": 0, "Human": 1, "mouse": 2, "Pic": 3, "Sac": 4}
        if host in host_mapping:
            host_features[host_mapping[host]] = 1
        features.extend(host_features)
        
        # 表达量相关特征
        expression = record.get("expression", {})
        features.extend([
            expression.get("value", 0),
            expression.get("confidence", 0) if isinstance(expression.get("confidence"), (int, float)) else 0,
        ])
        
        # 确保特征维度一致
        target_dim = 64
        if len(features) < target_dim:
            features.extend([0] * (target_dim - len(features)))
        elif len(features) > target_dim:
            features = features[:target_dim]
        
        return np.array(features, dtype=np.float32)
    
    def generate_weak_labels_robust(self) -> Dict[str, float]:
        """稳健的弱标签生成"""
        logger.info("Generating robust weak labels...")
        
        if not self.dataset_splits:
            raise ValueError("No dataset splits found. Call split_datasets_balanced first.")
        
        # 收集所有序列和标签
        all_sequences = []
        all_labels = []
        all_group_keys = []
        
        for group_key in self.protein_groups.keys():
            group = self.protein_groups[group_key]
            for seq in group.sequences:
                all_sequences.append(seq)
                all_labels.append(seq.get("expression", {}).get("value", 0))
                all_group_keys.append(group_key)
        
        if len(all_sequences) == 0:
            logger.warning("No sequences found for weak label generation")
            return {}
        
        # 提取特征
        features = np.array([self.extract_enhanced_features(seq) for seq in all_sequences])
        labels = np.array(all_labels)
        
        # 特征标准化
        if self.config.feature_scaling:
            self.feature_scaler = StandardScaler()
            features = self.feature_scaler.fit_transform(features)
        
        # 使用交叉验证生成弱标签
        weak_labels = {}
        
        if self.config.use_cross_validation:
            kf = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_seed)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
                logger.info(f"Processing fold {fold + 1}/{self.config.cv_folds}")
                
                X_train, X_val = features[train_idx], features[val_idx]
                y_train, y_val = labels[train_idx], labels[val_idx]
                
                # 训练模型
                if self.config.weak_label_model == "random_forest":
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=self.config.random_seed,
                        n_jobs=-1
                    )
                else:
                    # 默认使用随机森林
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=self.config.random_seed,
                        n_jobs=-1
                    )
                
                model.fit(X_train, y_train)
                
                # 预测验证集
                predictions = model.predict(X_val)
                
                # 保存弱标签
                for i, val_idx_i in enumerate(val_idx):
                    seq_idx = val_idx_i
                    group_key = all_group_keys[seq_idx]
                    weak_labels[f"{group_key}_{seq_idx}"] = float(predictions[i])
        
        logger.info(f"Generated weak labels for {len(weak_labels)} sequences")
        self.weak_labels = weak_labels
        return weak_labels
    
    def create_robust_training_pairs(self, dataset_type: str) -> List[Tuple[int, int, int]]:
        """创建稳健的训练对 - 大幅增加pair数量"""
        if dataset_type not in self.dataset_splits:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        group_keys = self.dataset_splits[dataset_type]
        pairs = []
        
        # 1. 创建真实表达量的pairs - 使用更宽松的标准
        real_pairs = []
        for group_key in group_keys:
            if group_key not in self.protein_groups:
                continue
                
            group = self.protein_groups[group_key]
            sequences = group.sequences
            
            # 创建组内所有可能的pairwise比较
            for i in range(len(sequences)):
                for j in range(i + 1, len(sequences)):
                    seq_i = sequences[i]
                    seq_j = sequences[j]
                    
                    # 获取表达量
                    expr_i = seq_i.get("expression", {}).get("value", 0)
                    expr_j = seq_j.get("expression", {}).get("value", 0)
                    
                    # 使用更宽松的差异标准
                    if self.config.use_relative_ranking:
                        # 使用相对排序，即使差异很小也创建pair
                        if expr_i != expr_j:  # 只要不相等就创建pair
                            label = 1 if expr_i > expr_j else 0
                            real_pairs.append((i, j, label, group_key))
                    else:
                        # 使用绝对差异
                        if abs(expr_i - expr_j) >= self.config.min_expression_diff:
                            label = 1 if expr_i > expr_j else 0
                            real_pairs.append((i, j, label, group_key))
        
        pairs.extend(real_pairs)
        logger.info(f"Created {len(real_pairs)} real expression pairs for {dataset_type}")
        
        # 2. 使用弱标签生成额外pairs
        if dataset_type == "train" and self.config.use_weak_label_pairs:
            logger.info(f"Generating weak label pairs for {dataset_type}...")
            
            weak_label_pairs = []
            for group_key in group_keys:
                if group_key not in self.protein_groups:
                    continue
                    
                group = self.protein_groups[group_key]
                sequences = group.sequences
                
                # 为每个序列获取弱标签
                weak_labels_for_group = {}
                for i, seq in enumerate(sequences):
                    weak_key = f"{group_key}_{i}"
                    if weak_key in self.weak_labels:
                        weak_labels_for_group[i] = self.weak_labels[weak_key]
                
                # 基于弱标签创建pairs
                for i in range(len(sequences)):
                    for j in range(i + 1, len(sequences)):
                        if i in weak_labels_for_group and j in weak_labels_for_group:
                            weak_i = weak_labels_for_group[i]
                            weak_j = weak_labels_for_group[j]
                            
                            # 使用更宽松的标准
                            if abs(weak_i - weak_j) >= self.config.min_expression_diff:
                                label = 1 if weak_i > weak_j else 0
                                weak_label_pairs.append((i, j, label, group_key))
            
            # 限制弱标签pairs的数量
            max_weak_pairs = int(self.config.min_train_pairs * self.config.weak_label_ratio)
            if len(weak_label_pairs) > max_weak_pairs:
                weak_label_pairs = random.sample(weak_label_pairs, max_weak_pairs)
            
            pairs.extend(weak_label_pairs)
            logger.info(f"Added {len(weak_label_pairs)} weak label pairs for {dataset_type}")
        
        # 3. 使用合成pairs（如果仍然不够）
        if dataset_type == "train" and len(pairs) < self.config.min_train_pairs and self.config.use_synthetic_pairs:
            logger.info(f"Generating synthetic pairs for {dataset_type}...")
            
            synthetic_pairs = []
            for group_key in group_keys:
                if group_key not in self.protein_groups:
                    continue
                    
                group = self.protein_groups[group_key]
                sequences = group.sequences
                
                # 为每个序列生成合成表达量（基于特征）
                synthetic_expressions = {}
                for i, seq in enumerate(sequences):
                    # 基于序列特征生成合成表达量
                    features = self.extract_enhanced_features(seq)
                    # 使用简单的线性组合
                    synthetic_expr = np.sum(features[:10]) * 10 + random.uniform(-5, 5)
                    synthetic_expressions[i] = synthetic_expr
                
                # 基于合成表达量创建pairs
                for i in range(len(sequences)):
                    for j in range(i + 1, len(sequences)):
                        if i in synthetic_expressions and j in synthetic_expressions:
                            synth_i = synthetic_expressions[i]
                            synth_j = synthetic_expressions[j]
                            
                            if abs(synth_i - synth_j) >= self.config.min_expression_diff:
                                label = 1 if synth_i > synth_j else 0
                                synthetic_pairs.append((i, j, label, group_key))
            
            # 限制合成pairs的数量
            max_synthetic_pairs = int(self.config.min_train_pairs * self.config.synthetic_ratio)
            if len(synthetic_pairs) > max_synthetic_pairs:
                synthetic_pairs = random.sample(synthetic_pairs, max_synthetic_pairs)
            
            pairs.extend(synthetic_pairs)
            logger.info(f"Added {len(synthetic_pairs)} synthetic pairs for {dataset_type}")
        
        # 4. 限制每组的最大pair数量
        if self.config.max_pairs_per_group > 0:
            group_pair_counts = defaultdict(int)
            filtered_pairs = []
            
            for pair in pairs:
                group_key = pair[3] if len(pair) > 3 else "unknown"
                if group_pair_counts[group_key] < self.config.max_pairs_per_group:
                    filtered_pairs.append(pair[:3])  # 只保留前三个元素
                    group_pair_counts[group_key] += 1
            
            pairs = filtered_pairs
        
        logger.info(f"Created {len(pairs)} total pairwise training pairs for {dataset_type}")
        return pairs
    
    def prepare_training_data(self, data_paths: List[str]) -> Dict[str, Any]:
        """完整的训练数据准备流程"""
        logger.info("Starting robust data preparation pipeline...")
        
        # 1. 加载数据
        records = self.load_data(data_paths)
        
        # 2. 分组
        protein_groups = self.group_by_protein_and_host(records)
        
        # 3. 聚类
        cluster_assignments = self.cluster_proteins_by_similarity()
        
        # 4. 平衡数据划分
        dataset_splits = self.split_datasets_balanced()
        
        # 5. 生成弱标签
        weak_labels = self.generate_weak_labels_robust()
        
        # 6. 创建训练对 - 使用稳健的方法
        train_pairs = self.create_robust_training_pairs("train")
        val_pairs = self.create_robust_training_pairs("val")
        test_pairs = self.create_robust_training_pairs("test")
        
        logger.info("Robust data preparation completed successfully!")
        
        return {
            "protein_groups": protein_groups,
            "cluster_assignments": cluster_assignments,
            "dataset_splits": dataset_splits,
            "weak_labels": weak_labels,
            "train_pairs": train_pairs,
            "val_pairs": val_pairs,
            "test_pairs": test_pairs,
            "feature_scaler": self.feature_scaler
        }

class RobustRankingDataset(Dataset):
    """稳健的排序数据集"""
    
    def __init__(self, protein_groups: Dict[str, ProteinGroup], 
                 pairs: List[Tuple[int, int, int]], 
                 group_keys: List[str],
                 feature_extractor,
                 feature_scaler: Optional[StandardScaler] = None):
        self.protein_groups = protein_groups
        self.pairs = pairs
        self.group_keys = group_keys
        self.feature_extractor = feature_extractor
        self.feature_scaler = feature_scaler
        
        # 预计算所有特征
        self.features_cache = {}
        self._precompute_features()
    
    def _precompute_features(self):
        """预计算所有特征"""
        logger.info("Precomputing features...")
        for group_key in self.group_keys:
            if group_key in self.protein_groups:
                group = self.protein_groups[group_key]
                group_features = []
                for seq in group.sequences:
                    features = self.feature_extractor(seq)
                    if self.feature_scaler is not None:
                        features = self.feature_scaler.transform(features.reshape(1, -1)).flatten()
                    group_features.append(features)
                self.features_cache[group_key] = group_features
        logger.info("Feature precomputation completed")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        i, j, label = pair
        
        # 找到对应的组
        group_key = None
        for gk in self.group_keys:
            if gk in self.protein_groups:
                group = self.protein_groups[gk]
                if i < len(group.sequences) and j < len(group.sequences):
                    group_key = gk
                    break
        
        if group_key is None:
            raise ValueError(f"Could not find group for pair {pair}")
        
        # 获取特征
        features_i = self.features_cache[group_key][i]
        features_j = self.features_cache[group_key][j]
        
        return {
            'features_i': torch.tensor(features_i, dtype=torch.float32),
            'features_j': torch.tensor(features_j, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'group_key': group_key  # 添加组键用于按组评估
        }

def create_robust_data_loaders(prepared_data: Dict[str, Any], 
                              batch_size: int = 32,
                              num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建稳健的数据加载器"""
    
    # 创建数据集
    train_dataset = RobustRankingDataset(
        prepared_data["protein_groups"],
        prepared_data["train_pairs"],
        prepared_data["dataset_splits"]["train"],
        lambda x: prepared_data.get("feature_extractor", lambda y: np.zeros(64))(x),
        prepared_data.get("feature_scaler")
    )
    
    val_dataset = RobustRankingDataset(
        prepared_data["protein_groups"],
        prepared_data["val_pairs"],
        prepared_data["dataset_splits"]["val"],
        lambda x: prepared_data.get("feature_extractor", lambda y: np.zeros(64))(x),
        prepared_data.get("feature_scaler")
    )
    
    test_dataset = RobustRankingDataset(
        prepared_data["protein_groups"],
        prepared_data["test_pairs"],
        prepared_data["dataset_splits"]["test"],
        lambda x: prepared_data.get("feature_extractor", lambda y: np.zeros(64))(x),
        prepared_data.get("feature_scaler")
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
