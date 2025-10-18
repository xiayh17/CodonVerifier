"""
排序模型数据预处理模块

实现无泄漏训练的数据准备流程：
1. 按protein_id和host分组
2. 基于序列相似性聚类划分训练集/验证集/测试集
3. 构建排序训练对（组内pairwise生成）
4. 生成弱标签（K折交叉拟合）
5. 训练集格式转换（PyTorch可用格式）
"""

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RankingDataConfig:
    """排序数据预处理配置"""
    # 数据分组
    min_sequences_per_group: int = 2  # 每组最少序列数
    max_sequences_per_group: int = 1000  # 每组最多序列数
    
    # 聚类参数
    clustering_method: str = "agglomerative"  # 聚类方法
    n_clusters: Optional[int] = None  # 聚类数量，None表示自动确定
    similarity_threshold: float = 0.8  # 序列相似性阈值
    
    # 数据集划分
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # 弱标签生成
    k_folds: int = 5  # K折交叉验证
    weak_label_model: str = "random_forest"  # 弱标签模型类型
    
    # 特征处理
    feature_scaling: bool = True
    feature_selection: bool = False
    
    # 随机种子
    random_seed: int = 42


class ProteinGroup:
    """蛋白质组类，包含相同protein_id和host的所有序列"""
    
    def __init__(self, protein_id: str, host: str, sequences: List[Dict]):
        self.protein_id = protein_id
        self.host = host
        self.sequences = sequences
        self.aa_sequence = sequences[0].get("protein_aa", "") if sequences else ""
        self.group_key = f"{protein_id}_{host}"
        
    def __len__(self):
        return len(self.sequences)
    
    def get_expression_values(self) -> List[float]:
        """获取组内所有序列的表达值"""
        values = []
        for seq in self.sequences:
            expr = seq.get("expression", {})
            if isinstance(expr, dict):
                value = expr.get("value", 0.0)
            else:
                value = float(expr) if expr is not None else 0.0
            values.append(value)
        return values
    
    def get_sequences(self) -> List[str]:
        """获取组内所有DNA序列"""
        return [seq.get("sequence", "") for seq in self.sequences]


class RankingDataProcessor:
    """排序数据处理器"""
    
    def __init__(self, config: Optional[RankingDataConfig] = None):
        self.config = config or RankingDataConfig()
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        self.protein_groups: Dict[str, ProteinGroup] = {}
        self.cluster_assignments: Dict[str, int] = {}
        self.dataset_splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        self.weak_labels: Dict[str, float] = {}
        self.feature_scaler: Optional[StandardScaler] = None
        
    def load_data(self, data_paths: List[str]) -> List[Dict]:
        """加载JSONL数据文件"""
        all_records = []
        
        for path in data_paths:
            logger.info(f"Loading data from {path}")
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        all_records.append(record)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line: {e}")
        
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
                
            # 使用氨基酸序列作为分组键，而不是protein_id
            group_key = f"{protein_aa}_{host}"
            groups[group_key].append(record)
        
        logger.info(f"Found {len(groups)} raw groups before filtering")
        
        # 创建ProteinGroup对象并过滤
        protein_groups = {}
        for group_key, sequences in groups.items():
            if len(sequences) >= self.config.min_sequences_per_group:
                if len(sequences) <= self.config.max_sequences_per_group:
                    # 从第一个序列获取protein_id作为代表
                    protein_id = sequences[0].get("protein_id", "unknown")
                    host = sequences[0].get("host", "unknown")
                    protein_groups[group_key] = ProteinGroup(protein_id, host, sequences)
                else:
                    # 如果序列太多，随机采样
                    sampled = random.sample(sequences, self.config.max_sequences_per_group)
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
    
    def compute_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """计算两个氨基酸序列的相似性"""
        if not seq1 or not seq2:
            return 0.0
        
        # 简单的序列比对相似性计算
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0
        
        matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
        return matches / min_len
    
    def cluster_proteins_by_similarity(self) -> Dict[str, int]:
        """基于序列相似性对蛋白质进行聚类"""
        logger.info("Clustering proteins by sequence similarity...")
        
        if not self.protein_groups:
            raise ValueError("No protein groups found. Call group_by_protein_and_host first.")
        
        # 提取氨基酸序列
        group_keys = list(self.protein_groups.keys())
        aa_sequences = [self.protein_groups[key].aa_sequence for key in group_keys]
        
        # 计算相似性矩阵
        n_groups = len(group_keys)
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
        
        # 聚类
        if self.config.n_clusters is None:
            # 自动确定聚类数量
            n_clusters = max(2, min(10, n_groups // 5))
        else:
            n_clusters = self.config.n_clusters
        
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
    
    def split_datasets_by_cluster(self) -> Dict[str, List[str]]:
        """基于聚类结果划分训练集/验证集/测试集"""
        logger.info("Splitting datasets by cluster...")
        
        if not self.cluster_assignments:
            raise ValueError("No cluster assignments found. Call cluster_proteins_by_similarity first.")
        
        # 按聚类分组
        clusters = defaultdict(list)
        for group_key, cluster_id in self.cluster_assignments.items():
            clusters[cluster_id].append(group_key)
        
        # 随机打乱聚类顺序
        cluster_ids = list(clusters.keys())
        random.shuffle(cluster_ids)
        
        # 按比例分配聚类到不同数据集
        n_clusters = len(cluster_ids)
        n_train = int(n_clusters * self.config.train_ratio)
        n_val = int(n_clusters * self.config.val_ratio)
        
        train_clusters = cluster_ids[:n_train]
        val_clusters = cluster_ids[n_train:n_train + n_val]
        test_clusters = cluster_ids[n_train + n_val:]
        
        # 分配组到数据集
        train_groups = []
        val_groups = []
        test_groups = []
        
        for cluster_id in train_clusters:
            train_groups.extend(clusters[cluster_id])
        
        for cluster_id in val_clusters:
            val_groups.extend(clusters[cluster_id])
        
        for cluster_id in test_clusters:
            test_groups.extend(clusters[cluster_id])
        
        self.dataset_splits = {
            "train": train_groups,
            "val": val_groups,
            "test": test_groups
        }
        
        logger.info(f"Dataset splits:")
        logger.info(f"  Train: {len(train_groups)} groups")
        logger.info(f"  Val: {len(val_groups)} groups")
        logger.info(f"  Test: {len(test_groups)} groups")
        
        return self.dataset_splits
    
    def extract_features(self, record: Dict) -> np.ndarray:
        """从记录中提取特征向量"""
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
        
        # Evo2特征
        features.extend([
            record.get("evo2_loglik", 0),
            record.get("evo2_avg_loglik", 0),
            record.get("evo2_perplexity", 0),
            record.get("evo2_geom", 0),
            record.get("evo2_bpb", 0),
            record.get("evo2_delta_bpb", 0),
            record.get("evo2_ref_bpb", 0),
            record.get("evo2_delta_nll", 0),
        ])
        
        # 密码子特征
        features.extend([
            record.get("codon_gc", 0),
            record.get("codon_cpb", 0),
            record.get("codon_cpg_count", 0),
            record.get("codon_cpg_freq", 0),
            record.get("codon_cpg_obs_exp", 0),
            record.get("codon_upa_count", 0),
            record.get("codon_upa_freq", 0),
            record.get("codon_upa_obs_exp", 0),
            record.get("codon_rare_runs", 0),
            record.get("codon_rare_run_total_len", 0),
            record.get("codon_homopolymers", 0),
            record.get("codon_homopoly_total_len", 0),
        ])
        
        # 结构特征
        struct_features = record.get("struct_plddt_mean", 0)
        features.extend([
            record.get("struct_plddt_mean", 0),
            record.get("struct_plddt_min", 0),
            record.get("struct_plddt_max", 0),
            record.get("struct_plddt_std", 0),
            record.get("struct_plddt_q25", 0),
            record.get("struct_plddt_q75", 0),
            record.get("struct_disorder_ratio", 0),
            record.get("struct_flexible_ratio", 0),
            record.get("struct_sasa_mean", 0),
            record.get("struct_sasa_total", 0),
            record.get("struct_sasa_polar_ratio", 0),
            record.get("struct_helix_ratio", 0),
            record.get("struct_sheet_ratio", 0),
            record.get("struct_coil_ratio", 0),
            record.get("struct_has_signal_peptide", 0),
            record.get("struct_has_transmembrane", 0),
            record.get("struct_tm_helix_count", 0),
            record.get("struct_pae_available", 0),
        ])
        
        # 上下文特征
        features.extend([
            record.get("ctx_promoter_strength", 0),
            record.get("ctx_rbs_strength", 0),
            record.get("ctx_rbs_spacing", 0),
            record.get("ctx_kozak_score", 0),
            record.get("ctx_vector_copy_number", 0),
            record.get("ctx_has_selection_marker", 0),
            record.get("ctx_temperature_norm", 0),
            record.get("ctx_inducer_concentration", 0),
            record.get("ctx_growth_phase_score", 0),
            record.get("ctx_localization_score", 0),
        ])
        
        return np.array(features, dtype=np.float32)
    
    def generate_weak_labels(self) -> Dict[str, float]:
        """使用K折交叉验证生成弱标签"""
        logger.info("Generating weak labels using K-fold cross-validation...")
        
        # 收集所有序列的特征和表达值
        all_features = []
        all_expressions = []
        all_sequence_ids = []
        
        for group_key, group in self.protein_groups.items():
            for i, sequence in enumerate(group.sequences):
                features = self.extract_features(sequence)
                expr = sequence.get("expression", {})
                if isinstance(expr, dict):
                    value = expr.get("value", 0.0)
                else:
                    value = float(expr) if expr is not None else 0.0
                
                all_features.append(features)
                all_expressions.append(value)
                all_sequence_ids.append(f"{group_key}_{i}")
        
        all_features = np.array(all_features)
        all_expressions = np.array(all_expressions)
        
        # 特征标准化
        if self.config.feature_scaling:
            self.feature_scaler = StandardScaler()
            all_features = self.feature_scaler.fit_transform(all_features)
        
        # K折交叉验证生成弱标签
        kf = KFold(n_splits=self.config.k_folds, shuffle=True, random_state=self.config.random_seed)
        weak_labels = {}
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(all_features)):
            logger.info(f"Processing fold {fold + 1}/{self.config.k_folds}")
            
            # 训练弱标签模型
            X_train = all_features[train_idx]
            y_train = all_expressions[train_idx]
            
            if self.config.weak_label_model == "random_forest":
                model = RandomForestRegressor(n_estimators=100, random_state=self.config.random_seed)
            else:
                raise ValueError(f"Unknown weak label model: {self.config.weak_label_model}")
            
            model.fit(X_train, y_train)
            
            # 预测验证集
            X_val = all_features[val_idx]
            predictions = model.predict(X_val)
            
            # 保存弱标签
            for i, seq_id in enumerate([all_sequence_ids[idx] for idx in val_idx]):
                weak_labels[seq_id] = float(predictions[i])
        
        self.weak_labels = weak_labels
        logger.info(f"Generated weak labels for {len(weak_labels)} sequences")
        
        return weak_labels
    
    def create_pairwise_training_pairs(self, dataset_split: str) -> List[Tuple[int, int, int]]:
        """为指定数据集创建pairwise训练对"""
        logger.info(f"Creating pairwise training pairs for {dataset_split} dataset...")
        
        if dataset_split not in self.dataset_splits:
            raise ValueError(f"Unknown dataset split: {dataset_split}")
        
        group_keys = self.dataset_splits[dataset_split]
        pairs = []
        
        for group_key in group_keys:
            if group_key not in self.protein_groups:
                continue
            
            group = self.protein_groups[group_key]
            sequences = group.sequences
            expressions = group.get_expression_values()
            
            # 为组内每对序列创建比较对
            for i in range(len(sequences)):
                for j in range(i + 1, len(sequences)):
                    seq_id_i = f"{group_key}_{i}"
                    seq_id_j = f"{group_key}_{j}"
                    
                    # 使用弱标签进行比较
                    weak_label_i = self.weak_labels.get(seq_id_i, 0.0)
                    weak_label_j = self.weak_labels.get(seq_id_j, 0.0)
                    
                    # 创建比较对：如果i的表达高于j，标签为1，否则为0
                    if weak_label_i > weak_label_j:
                        pairs.append((i, j, 1))
                    elif weak_label_i < weak_label_j:
                        pairs.append((i, j, 0))
                    # 如果相等，跳过这个对
        
        logger.info(f"Created {len(pairs)} pairwise training pairs for {dataset_split}")
        return pairs
    
    def prepare_training_data(self, data_paths: List[str]) -> Dict[str, Any]:
        """完整的数据准备流程"""
        logger.info("Starting complete data preparation pipeline...")
        
        # 1. 加载数据
        records = self.load_data(data_paths)
        
        # 2. 按蛋白质和宿主分组
        self.group_by_protein_and_host(records)
        
        # 3. 基于序列相似性聚类
        self.cluster_proteins_by_similarity()
        
        # 4. 划分数据集
        self.split_datasets_by_cluster()
        
        # 5. 生成弱标签
        self.generate_weak_labels()
        
        # 6. 创建训练对
        train_pairs = self.create_pairwise_training_pairs("train")
        val_pairs = self.create_pairwise_training_pairs("val")
        test_pairs = self.create_pairwise_training_pairs("test")
        
        # 准备返回数据
        result = {
            "protein_groups": self.protein_groups,
            "cluster_assignments": self.cluster_assignments,
            "dataset_splits": self.dataset_splits,
            "weak_labels": self.weak_labels,
            "train_pairs": train_pairs,
            "val_pairs": val_pairs,
            "test_pairs": test_pairs,
            "feature_scaler": self.feature_scaler,
        }
        
        logger.info("Data preparation completed successfully!")
        return result


class RankingDataset(Dataset):
    """PyTorch数据集类，用于排序模型训练"""
    
    def __init__(self, protein_groups: Dict[str, ProteinGroup], 
                 pairs: List[Tuple[int, int, int]], 
                 weak_labels: Dict[str, float],
                 feature_scaler: Optional[StandardScaler] = None):
        self.protein_groups = protein_groups
        self.pairs = pairs
        self.weak_labels = weak_labels
        self.feature_scaler = feature_scaler
        
        # 构建序列索引映射
        self.sequence_index_map = {}
        self.features_cache = {}
        
        idx = 0
        for group_key, group in protein_groups.items():
            for i, sequence in enumerate(group.sequences):
                seq_id = f"{group_key}_{i}"
                self.sequence_index_map[idx] = (group_key, i)
                
                # 提取并缓存特征
                features = self._extract_features(sequence)
                if self.feature_scaler is not None:
                    features = self.feature_scaler.transform(features.reshape(1, -1)).flatten()
                self.features_cache[idx] = torch.tensor(features, dtype=torch.float32)
                
                idx += 1
    
    def _extract_features(self, record: Dict) -> np.ndarray:
        """提取特征向量（与RankingDataProcessor中的方法相同）"""
        features = []
        
        # 基础序列特征
        sequence = record.get("sequence", "")
        features.extend([
            len(sequence),
            sequence.count('A') / len(sequence) if sequence else 0,
            sequence.count('T') / len(sequence) if sequence else 0,
            sequence.count('G') / len(sequence) if sequence else 0,
            sequence.count('C') / len(sequence) if sequence else 0,
        ])
        
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
        
        # Evo2特征
        features.extend([
            record.get("evo2_loglik", 0),
            record.get("evo2_avg_loglik", 0),
            record.get("evo2_perplexity", 0),
            record.get("evo2_geom", 0),
            record.get("evo2_bpb", 0),
            record.get("evo2_delta_bpb", 0),
            record.get("evo2_ref_bpb", 0),
            record.get("evo2_delta_nll", 0),
        ])
        
        # 密码子特征
        features.extend([
            record.get("codon_gc", 0),
            record.get("codon_cpb", 0),
            record.get("codon_cpg_count", 0),
            record.get("codon_cpg_freq", 0),
            record.get("codon_cpg_obs_exp", 0),
            record.get("codon_upa_count", 0),
            record.get("codon_upa_freq", 0),
            record.get("codon_upa_obs_exp", 0),
            record.get("codon_rare_runs", 0),
            record.get("codon_rare_run_total_len", 0),
            record.get("codon_homopolymers", 0),
            record.get("codon_homopoly_total_len", 0),
        ])
        
        # 结构特征
        features.extend([
            record.get("struct_plddt_mean", 0),
            record.get("struct_plddt_min", 0),
            record.get("struct_plddt_max", 0),
            record.get("struct_plddt_std", 0),
            record.get("struct_plddt_q25", 0),
            record.get("struct_plddt_q75", 0),
            record.get("struct_disorder_ratio", 0),
            record.get("struct_flexible_ratio", 0),
            record.get("struct_sasa_mean", 0),
            record.get("struct_sasa_total", 0),
            record.get("struct_sasa_polar_ratio", 0),
            record.get("struct_helix_ratio", 0),
            record.get("struct_sheet_ratio", 0),
            record.get("struct_coil_ratio", 0),
            record.get("struct_has_signal_peptide", 0),
            record.get("struct_has_transmembrane", 0),
            record.get("struct_tm_helix_count", 0),
            record.get("struct_pae_available", 0),
        ])
        
        # 上下文特征
        features.extend([
            record.get("ctx_promoter_strength", 0),
            record.get("ctx_rbs_strength", 0),
            record.get("ctx_rbs_spacing", 0),
            record.get("ctx_kozak_score", 0),
            record.get("ctx_vector_copy_number", 0),
            record.get("ctx_has_selection_marker", 0),
            record.get("ctx_temperature_norm", 0),
            record.get("ctx_inducer_concentration", 0),
            record.get("ctx_growth_phase_score", 0),
            record.get("ctx_localization_score", 0),
        ])
        
        return np.array(features, dtype=np.float32)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        i, j, label = self.pairs[idx]
        
        # 获取特征
        features_i = self.features_cache[i]
        features_j = self.features_cache[j]
        
        return features_i, features_j, torch.tensor(label, dtype=torch.float32)


def create_data_loaders(prepared_data: Dict[str, Any], 
                       batch_size: int = 32,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练、验证和测试数据加载器"""
    
    # 创建数据集
    train_dataset = RankingDataset(
        prepared_data["protein_groups"],
        prepared_data["train_pairs"],
        prepared_data["weak_labels"],
        prepared_data["feature_scaler"]
    )
    
    val_dataset = RankingDataset(
        prepared_data["protein_groups"],
        prepared_data["val_pairs"],
        prepared_data["weak_labels"],
        prepared_data["feature_scaler"]
    )
    
    test_dataset = RankingDataset(
        prepared_data["protein_groups"],
        prepared_data["test_pairs"],
        prepared_data["weak_labels"],
        prepared_data["feature_scaler"]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def main():
    """主函数，演示数据预处理流程"""
    # 配置
    config = RankingDataConfig(
        min_sequences_per_group=2,
        max_sequences_per_group=100,
        k_folds=5,
        random_seed=42
    )
    
    # 数据路径
    data_paths = [
        "/mnt/c/Users/xiayh17/Documents/GitHub/CodonVerifier/data/weaklabels/Ec_complete_v3_updated.jsonl",
        "/mnt/c/Users/xiayh17/Documents/GitHub/CodonVerifier/data/weaklabels/Human_complete_v3_updated.jsonl",
        "/mnt/c/Users/xiayh17/Documents/GitHub/CodonVerifier/data/weaklabels/mouse_complete_v3_updated.jsonl",
        "/mnt/c/Users/xiayh17/Documents/GitHub/CodonVerifier/data/weaklabels/Pic_complete_v3_updated.jsonl",
        "/mnt/c/Users/xiayh17/Documents/GitHub/CodonVerifier/data/weaklabels/Sac_complete_v3_updated.jsonl",
    ]
    
    # 创建处理器
    processor = RankingDataProcessor(config)
    
    # 执行数据准备
    prepared_data = processor.prepare_training_data(data_paths)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(prepared_data)
    
    logger.info("Data preparation completed successfully!")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    return prepared_data, train_loader, val_loader, test_loader


if __name__ == "__main__":
    main()
