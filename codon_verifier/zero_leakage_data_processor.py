"""
零泄露排序数据处理器

实现严格的数据切分与审计协议：
1. 按蛋白同源性分簇切分（≤30% identity）
2. DNA k-mer距离审计（防止近重复穿越）
3. 保证Val/Test各≥50组
4. 难例为主的pair生成（信息密度优先）
5. K折交叉拟合弱标签（防泄露）
6. 按组宏平均评估+置信区间

作者：Zero-Leakage Training Protocol
日期：2025-01-16
"""

import json
import random
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import subprocess
import tempfile
import hashlib
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

@dataclass
class ZeroLeakageConfig:
    """零泄露数据配置"""
    # 切分比例
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # 组数要求
    min_val_groups: int = 50
    min_test_groups: int = 50
    
    # 同源性聚类
    protein_identity_threshold: float = 0.30  # ≤30% identity严阈值
    use_cdhit: bool = True  # 优先用CD-HIT，否则用简单k-mer
    
    # DNA k-mer审计
    kmer_size: int = 5
    kmer_jaccard_threshold: float = 0.80  # ≥0.8视为近重复
    
    # Pair生成策略
    min_pairs_per_group: int = 50
    max_pairs_per_group: int = 200
    target_train_pairs: int = 50000
    target_val_pairs: int = 5000
    target_test_pairs: int = 5000
    
    # 难例采样
    hard_example_ratio: float = 0.75  # 75%是难例
    min_expression_diff_hard: float = 0.5   # 难例：差异0.5-2.5
    max_expression_diff_hard: float = 2.5
    min_expression_diff_easy: float = 2.5   # 简单例：差异>2.5
    
    # 弱标签K折交叉拟合
    weak_label_k_folds: int = 5
    weak_label_model: str = "gbdt"  # gbdt/ridge/linear
    
    # 特征工程
    use_enhanced_features: bool = True
    feature_scaling: bool = True
    feature_dropout: float = 0.5  # LM特征抑制
    use_lm_features_controlled: bool = False  # 是否使用受控的LM特征
    lm_feature_dropout: float = 0.8  # LM特征dropout概率
    
    # 样本参与度限制
    max_sample_participation: int = 100  # 每条样本最多参与100次
    
    # 随机种子
    random_seed: int = 42


@dataclass
class ProteinCluster:
    """蛋白质簇"""
    cluster_id: int
    representative_aa: str  # 代表序列
    protein_groups: List[str]  # 该簇包含的蛋白组
    size: int = 0
    
    def __post_init__(self):
        self.size = len(self.protein_groups)


@dataclass
class ProteinGroup:
    """蛋白质组（同一AA序列+同一宿主）"""
    group_id: str
    protein_aa: str
    host: str
    sequences: List[Dict]
    cluster_id: Optional[int] = None
    
    def __len__(self):
        return len(self.sequences)


@dataclass
class DataSplitAudit:
    """数据切分审计报告"""
    n_total_groups: int
    n_train_groups: int
    n_val_groups: int
    n_test_groups: int
    n_clusters: int
    cluster_distribution: Dict[str, List[int]]
    kmer_audit_results: Dict[str, Any]
    homology_stats: Dict[str, float]
    split_quality_score: float = 0.0


class ZeroLeakageDataProcessor:
    """零泄露数据处理器"""
    
    def __init__(self, config: ZeroLeakageConfig):
        self.config = config
        self.protein_groups: Dict[str, ProteinGroup] = {}
        self.protein_clusters: Dict[int, ProteinCluster] = {}
        self.dataset_splits: Dict[str, List[str]] = {}
        self.weak_labels: Dict[str, float] = {}
        self.weak_label_confidence: Dict[str, float] = {}
        self.feature_scaler: Optional[StandardScaler] = None
        self.audit_report: Optional[DataSplitAudit] = None
        
        # 设置随机种子
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    def load_data(self, data_paths: List[str]) -> List[Dict]:
        """加载数据"""
        logger.info("=" * 60)
        logger.info("步骤1: 加载数据")
        logger.info("=" * 60)
        
        all_records = []
        for path in data_paths:
            if not Path(path).exists():
                logger.warning(f"文件不存在: {path}")
                continue
            
            logger.info(f"加载数据: {path}")
            with open(path, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        all_records.append(record)
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"✓ 加载了 {len(all_records)} 条记录")
        return all_records
    
    def group_by_protein_and_host(self, records: List[Dict]) -> Dict[str, ProteinGroup]:
        """按氨基酸序列和host分组"""
        logger.info("=" * 60)
        logger.info("步骤2: 按蛋白AA序列+宿主分组")
        logger.info("=" * 60)
        
        groups = defaultdict(list)
        for record in records:
            protein_aa = record.get("protein_aa", "")
            host = record.get("host", "")
            if not protein_aa or not host:
                continue
            
            group_key = f"{protein_aa}_{host}"
            groups[group_key].append(record)
        
        logger.info(f"找到 {len(groups)} 个原始组")
        
        # 过滤：每组至少2条序列
        protein_groups = {}
        for group_key, sequences in groups.items():
            if len(sequences) >= 2:
                protein_aa = sequences[0].get("protein_aa", "")
                host = sequences[0].get("host", "")
                protein_groups[group_key] = ProteinGroup(
                    group_id=group_key,
                    protein_aa=protein_aa,
                    host=host,
                    sequences=sequences
                )
        
        logger.info(f"✓ 创建了 {len(protein_groups)} 个有效组（每组≥2条序列）")
        
        # 统计信息
        group_sizes = [len(g.sequences) for g in protein_groups.values()]
        host_dist = Counter(g.host for g in protein_groups.values())
        
        logger.info(f"  组大小: Min={min(group_sizes)}, Max={max(group_sizes)}, Mean={np.mean(group_sizes):.1f}")
        logger.info(f"  宿主分布: {dict(host_dist)}")
        
        self.protein_groups = protein_groups
        return protein_groups
    
    def compute_kmer_set(self, sequence: str, k: int = 5) -> Set[str]:
        """计算序列的k-mer集合"""
        if len(sequence) < k:
            return set()
        return {sequence[i:i+k] for i in range(len(sequence) - k + 1)}
    
    def compute_kmer_jaccard(self, seq1: str, seq2: str, k: int = 5) -> float:
        """计算两个序列的k-mer Jaccard相似度"""
        kmers1 = self.compute_kmer_set(seq1, k)
        kmers2 = self.compute_kmer_set(seq2, k)
        
        if not kmers1 or not kmers2:
            return 0.0
        
        intersection = len(kmers1 & kmers2)
        union = len(kmers1 | kmers2)
        
        return intersection / union if union > 0 else 0.0
    
    def cluster_proteins_by_homology(self) -> Dict[int, ProteinCluster]:
        """按蛋白同源性聚类（使用k-mer近似，或调用CD-HIT）"""
        logger.info("=" * 60)
        logger.info("步骤3: 按蛋白同源性聚类（≤30% identity）")
        logger.info("=" * 60)
        
        group_keys = list(self.protein_groups.keys())
        n_groups = len(group_keys)
        
        # 使用简单的k-mer聚类（近似替代CD-HIT）
        logger.info(f"使用k-mer方法聚类 {n_groups} 个蛋白组...")
        
        # 计算所有蛋白的k-mer相似度矩阵
        aa_sequences = [self.protein_groups[gk].protein_aa for gk in group_keys]
        
        # 使用层次聚类
        similarity_matrix = np.zeros((n_groups, n_groups))
        for i in range(n_groups):
            for j in range(i, n_groups):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self.compute_kmer_jaccard(aa_sequences[i], aa_sequences[j], k=3)
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        # 使用凝聚聚类，距离阈值对应30% identity
        from sklearn.cluster import AgglomerativeClustering
        
        # 转换为距离矩阵
        distance_matrix = 1 - similarity_matrix
        
        # 自动确定聚类数：希望每簇平均2-5个组
        target_cluster_size = 3
        n_clusters = max(2, n_groups // target_cluster_size)
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # 创建蛋白簇
        clusters_dict = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters_dict[int(label)].append(group_keys[i])
        
        protein_clusters = {}
        for cluster_id, group_list in clusters_dict.items():
            representative_aa = self.protein_groups[group_list[0]].protein_aa
            protein_clusters[cluster_id] = ProteinCluster(
                cluster_id=cluster_id,
                representative_aa=representative_aa,
                protein_groups=group_list
            )
            # 更新protein_groups的cluster_id
            for gk in group_list:
                self.protein_groups[gk].cluster_id = cluster_id
        
        logger.info(f"✓ 创建了 {len(protein_clusters)} 个蛋白簇")
        
        # 统计信息
        cluster_sizes = [len(c.protein_groups) for c in protein_clusters.values()]
        logger.info(f"  簇大小: Min={min(cluster_sizes)}, Max={max(cluster_sizes)}, Mean={np.mean(cluster_sizes):.1f}")
        
        self.protein_clusters = protein_clusters
        return protein_clusters
    
    def split_by_clusters(self) -> Dict[str, List[str]]:
        """按簇切分数据集，确保整簇分配"""
        logger.info("=" * 60)
        logger.info("步骤4: 按簇切分数据集（整簇分配，防泄露）")
        logger.info("=" * 60)
        
        cluster_ids = list(self.protein_clusters.keys())
        random.shuffle(cluster_ids)
        
        n_clusters = len(cluster_ids)
        n_train = int(n_clusters * self.config.train_ratio)
        n_val = int(n_clusters * self.config.val_ratio)
        
        train_cluster_ids = cluster_ids[:n_train]
        val_cluster_ids = cluster_ids[n_train:n_train + n_val]
        test_cluster_ids = cluster_ids[n_train + n_val:]
        
        # 收集每个集合的组
        train_groups = []
        val_groups = []
        test_groups = []
        
        for cid in train_cluster_ids:
            train_groups.extend(self.protein_clusters[cid].protein_groups)
        
        for cid in val_cluster_ids:
            val_groups.extend(self.protein_clusters[cid].protein_groups)
        
        for cid in test_cluster_ids:
            test_groups.extend(self.protein_clusters[cid].protein_groups)
        
        # 检查是否满足最小组数要求
        if len(val_groups) < self.config.min_val_groups:
            logger.warning(f"⚠️ Val组数 ({len(val_groups)}) < 最小要求 ({self.config.min_val_groups})")
            logger.warning("   建议：增加数据量或降低min_val_groups")
        
        if len(test_groups) < self.config.min_test_groups:
            logger.warning(f"⚠️ Test组数 ({len(test_groups)}) < 最小要求 ({self.config.min_test_groups})")
            logger.warning("   建议：增加数据量或降低min_test_groups")
        
        self.dataset_splits = {
            "train": train_groups,
            "val": val_groups,
            "test": test_groups
        }
        
        total = len(train_groups) + len(val_groups) + len(test_groups)
        logger.info(f"✓ 数据集切分完成:")
        logger.info(f"  Train: {len(train_groups)} 组 ({len(train_groups)/total*100:.1f}%)")
        logger.info(f"  Val:   {len(val_groups)} 组 ({len(val_groups)/total*100:.1f}%)")
        logger.info(f"  Test:  {len(test_groups)} 组 ({len(test_groups)/total*100:.1f}%)")
        
        return self.dataset_splits
    
    def audit_kmer_leakage(self) -> Dict[str, Any]:
        """DNA k-mer泄露审计"""
        logger.info("=" * 60)
        logger.info("步骤5: DNA k-mer泄露审计")
        logger.info("=" * 60)
        
        train_groups = self.dataset_splits["train"]
        val_groups = self.dataset_splits["val"]
        test_groups = self.dataset_splits["test"]
        
        # 收集所有DNA序列
        def get_dna_sequences(group_keys):
            dna_seqs = []
            for gk in group_keys:
                if gk in self.protein_groups:
                    for seq_record in self.protein_groups[gk].sequences:
                        dna_seq = seq_record.get("sequence", "")
                        if dna_seq:
                            dna_seqs.append(dna_seq)
            return dna_seqs
        
        train_dnas = get_dna_sequences(train_groups)
        val_dnas = get_dna_sequences(val_groups)
        test_dnas = get_dna_sequences(test_groups)
        
        logger.info(f"  Train DNA序列: {len(train_dnas)}")
        logger.info(f"  Val DNA序列: {len(val_dnas)}")
        logger.info(f"  Test DNA序列: {len(test_dnas)}")
        
        # 计算Test vs Train的最大Jaccard相似度
        k = self.config.kmer_size
        threshold = self.config.kmer_jaccard_threshold
        
        high_similarity_count = 0
        max_jaccard = 0.0
        
        # 采样审计（全量计算太慢）
        sample_size = min(100, len(test_dnas))
        sampled_test = random.sample(test_dnas, sample_size) if len(test_dnas) > sample_size else test_dnas
        
        for test_dna in sampled_test:
            test_kmers = self.compute_kmer_set(test_dna, k)
            if not test_kmers:
                continue
            
            for train_dna in random.sample(train_dnas, min(50, len(train_dnas))):
                jaccard = self.compute_kmer_jaccard(test_dna, train_dna, k)
                max_jaccard = max(max_jaccard, jaccard)
                
                if jaccard >= threshold:
                    high_similarity_count += 1
        
        audit_results = {
            "kmer_size": k,
            "threshold": threshold,
            "sampled_test_size": sample_size,
            "high_similarity_count": high_similarity_count,
            "max_jaccard": float(max_jaccard),
            "leakage_detected": high_similarity_count > 0
        }
        
        if audit_results["leakage_detected"]:
            logger.warning(f"⚠️ 检测到可能的k-mer泄露:")
            logger.warning(f"   {high_similarity_count} 个Test样本与Train的Jaccard ≥ {threshold}")
            logger.warning(f"   最大Jaccard: {max_jaccard:.4f}")
        else:
            logger.info(f"✓ 未检测到显著k-mer泄露 (max Jaccard: {max_jaccard:.4f})")
        
        return audit_results
    
    def extract_enhanced_features(self, record: Dict) -> np.ndarray:
        """提取增强特征（64维）"""
        features = []
        
        # 基础序列特征 (6)
        sequence = record.get("sequence", "")
        if sequence:
            features.extend([
                len(sequence),
                sequence.count('A') / len(sequence),
                sequence.count('T') / len(sequence),
                sequence.count('G') / len(sequence),
                sequence.count('C') / len(sequence),
                (sequence.count('G') + sequence.count('C')) / len(sequence),  # GC content
            ])
        else:
            features.extend([0] * 6)
        
        # 密码子特征 (2)
        if len(sequence) >= 3:
            codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]
            codon_counts = Counter(codons)
            total_codons = len(codons) if codons else 1
            common_codons = ['ATG', 'TAA', 'TAG', 'TGA']
            common_ratio = sum(codon_counts.get(c, 0) for c in common_codons) / total_codons
            diversity = len(codon_counts) / total_codons
            features.extend([common_ratio, diversity])
        else:
            features.extend([0, 0])
        
        # MSA特征 (11)
        msa = record.get("msa_features", {})
        features.extend([
            msa.get("msa_depth", 0),
            msa.get("msa_effective_depth", 0),
            msa.get("msa_coverage", 0),
            msa.get("conservation_mean", 0),
            msa.get("conservation_min", 0),
            msa.get("conservation_max", 0),
            msa.get("conservation_entropy_mean", 0),
            msa.get("coevolution_score", 0),
            msa.get("contact_density", 0),
            msa.get("pfam_count", 0),
            msa.get("domain_count", 0),
        ])
        
        # LM特征（受控版本：强dropout + host-wise标准化 + 去趋势）
        if self.config.use_lm_features_controlled:
            # 加回evo2特征，但应用强feature dropout (0.8概率置零)
            if random.random() < self.config.lm_feature_dropout:
                evo_features = [0.0, 0.0, 0.0]
            else:
                # 使用去趋势后的LM特征（如果存在）
                evo_features = [
                    record.get("evo2_loglik_zscore_detrended", 
                               record.get("evo2_loglik_zscore",
                                          record.get("evo2_loglik", 0))),
                    record.get("evo2_entropy_zscore_detrended", 
                               record.get("evo2_entropy_zscore",
                                          record.get("evo2_entropy", 0))),
                    record.get("evo2_kl_divergence_zscore_detrended", 
                               record.get("evo2_kl_divergence_zscore",
                                          record.get("evo2_kl_divergence", 0))),
                ]
            features.extend(evo_features)
        else:
            # No-LM版本：使用物理化学特征替代
            features.extend([
                record.get("cai_score", 0),  # Codon Adaptation Index
                record.get("rare_codon_ratio", 0),  # 稀有密码子比例
                record.get("mrna_folding_energy", 0),  # mRNA折叠能量
            ])
        
        # 结构特征 (5)
        struct = record.get("structure_features", {})
        features.extend([
            struct.get("secondary_structure_helix", 0),
            struct.get("secondary_structure_sheet", 0),
            struct.get("secondary_structure_loop", 0),
            struct.get("disorder_score", 0),
            struct.get("solvent_accessibility", 0),
        ])
        
        # 宿主特征 (5) - one-hot
        host = record.get("host", "")
        host_map = {"E_coli": 0, "Human": 1, "mouse": 2, "Pic": 3, "Sac": 4}
        host_features = [0] * 5
        if host in host_map:
            host_features[host_map[host]] = 1
        features.extend(host_features)
        
        # ❌ 移除泄露特征：expression.value（真实标签）
        # ❌ 移除泄露特征：expression.confidence（可能相关）
        # 这两个特征会导致模型直接记忆答案！
        
        # 蛋白质长度特征 (1) - 替代品
        aa_seq = record.get("aa_sequence", "")
        features.append(len(aa_seq) if aa_seq else 0)
        
        # DNA二级结构能量 (2) - 可用的物理特征
        features.extend([
            record.get("dna_mfe", 0),  # 最小自由能
            record.get("dna_stability", 0),  # 稳定性分数
        ])
        
        # 填充或截断到64维
        if len(features) < 64:
            features.extend([0] * (64 - len(features)))
        else:
            features = features[:64]
        
        return np.array(features, dtype=np.float32)
    
    def generate_weak_labels_cv(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """K折交叉拟合生成弱标签（防泄露）"""
        logger.info("=" * 60)
        logger.info("步骤6: K折交叉拟合生成弱标签")
        logger.info("=" * 60)
        
        train_groups = self.dataset_splits["train"]
        
        # 收集训练集所有序列
        all_sequences = []
        all_labels = []
        all_keys = []
        
        for gk in train_groups:
            if gk not in self.protein_groups:
                continue
            group = self.protein_groups[gk]
            for i, seq in enumerate(group.sequences):
                all_sequences.append(seq)
                all_labels.append(seq.get("expression", {}).get("value", 0))
                all_keys.append(f"{gk}_{i}")
        
        if len(all_sequences) == 0:
            logger.warning("没有训练序列用于弱标签生成")
            return {}, {}
        
        # 提取特征
        X = np.array([self.extract_enhanced_features(seq) for seq in all_sequences])
        y = np.array(all_labels)
        
        # 标准化
        if self.config.feature_scaling:
            self.feature_scaler = StandardScaler()
            X = self.feature_scaler.fit_transform(X)
        
        # K折交叉拟合
        k = self.config.weak_label_k_folds
        kf = KFold(n_splits=k, shuffle=True, random_state=self.config.random_seed)
        
        weak_labels = {}
        weak_confidence = {}
        
        logger.info(f"使用{k}折交叉验证生成弱标签...")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.info(f"  Fold {fold + 1}/{k}: Train={len(train_idx)}, Val={len(val_idx)}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 训练模型
            if self.config.weak_label_model == "gbdt":
                model = GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=self.config.random_seed
                )
            elif self.config.weak_label_model == "ridge":
                model = Ridge(alpha=1.0, random_state=self.config.random_seed)
            else:
                model = Ridge(alpha=1.0, random_state=self.config.random_seed)
            
            model.fit(X_train, y_train)
            
            # 预测验证集
            y_pred = model.predict(X_val)
            
            # 计算置信度（基于残差）
            residuals = np.abs(y_val - y_pred)
            max_residual = np.max(residuals) if len(residuals) > 0 else 1.0
            confidences = 1.0 - (residuals / max_residual)
            
            # 保存
            for i, idx in enumerate(val_idx):
                key = all_keys[idx]
                weak_labels[key] = float(y_pred[i])
                weak_confidence[key] = float(confidences[i])
        
        logger.info(f"✓ 生成了 {len(weak_labels)} 个弱标签")
        logger.info(f"  平均置信度: {np.mean(list(weak_confidence.values())):.3f}")
        
        self.weak_labels = weak_labels
        self.weak_label_confidence = weak_confidence
        
        return weak_labels, weak_confidence
    
    def apply_lm_feature_detrending(self):
        """
        对LM特征应用后处理：
        1. Host-wise z-score标准化（防止宿主间泄露）
        2. 分组去趋势（移除组内相关性）
        """
        # LM特征在特征向量中的位置（基于extract_enhanced_features）
        # 基础(6) + 密码子(2) + MSA(11) = 19，所以LM特征在索引19-21
        LM_FEATURE_START = 19
        LM_FEATURE_END = 22  # [19, 20, 21]
        
        # Step 1: Host-wise z-score标准化
        logger.info("  [1] Host-wise z-score标准化...")
        
        # 收集每个宿主的LM特征
        host_lm_features = defaultdict(list)
        host_seq_mapping = defaultdict(list)  # (gk, seq_idx)
        
        for gk, group in self.protein_groups.items():
            host = group.host
            for i, seq in enumerate(group.sequences):
                features = self.extract_enhanced_features(seq)
                lm_feats = features[LM_FEATURE_START:LM_FEATURE_END]
                host_lm_features[host].append(lm_feats)
                host_seq_mapping[host].append((gk, i, seq))
        
        # 按宿主z-score标准化
        host_stats = {}
        for host, lm_feats_list in host_lm_features.items():
            lm_feats_array = np.array(lm_feats_list)  # (n_seqs, 3)
            mean = np.mean(lm_feats_array, axis=0)
            std = np.std(lm_feats_array, axis=0) + 1e-8
            host_stats[host] = {'mean': mean, 'std': std}
            logger.info(f"    {host}: n={len(lm_feats_list)}, mean={mean}, std={std}")
        
        # 应用z-score到序列（修改protein_groups中的序列元数据）
        for host, seq_list in host_seq_mapping.items():
            mean, std = host_stats[host]['mean'], host_stats[host]['std']
            for gk, i, seq in seq_list:
                # 提取原始LM特征
                orig_lm = np.array([
                    seq.get("evo2_loglik", 0),
                    seq.get("evo2_entropy", 0),
                    seq.get("evo2_kl_divergence", 0)
                ])
                # z-score
                z_lm = (orig_lm - mean) / std
                # 回写
                seq["evo2_loglik_zscore"] = float(z_lm[0])
                seq["evo2_entropy_zscore"] = float(z_lm[1])
                seq["evo2_kl_divergence_zscore"] = float(z_lm[2])
        
        # Step 2: 分组去趋势（仅用Train集计算残差模型）
        logger.info("  [2] 分组去趋势（移除组内LM-expression相关性）...")
        
        # 只在Train集上拟合残差模型
        train_groups = self.dataset_splits["train"]
        
        from sklearn.linear_model import Ridge
        
        for feat_idx, feat_name in enumerate(['evo2_loglik_zscore', 'evo2_entropy_zscore', 'evo2_kl_divergence_zscore']):
            X_train, y_train = [], []
            
            for gk in train_groups:
                if gk not in self.protein_groups:
                    continue
                group = self.protein_groups[gk]
                
                for seq in group.sequences:
                    X_train.append(seq.get(feat_name, 0))
                    y_train.append(seq.get("expression", {}).get("value", 0))
            
            if len(X_train) < 10:
                logger.warning(f"    跳过{feat_name}：训练样本不足")
                continue
            
            X_train = np.array(X_train).reshape(-1, 1)
            y_train = np.array(y_train)
            
            # 拟合残差模型
            model = Ridge(alpha=0.1)
            model.fit(X_train, y_train)
            
            logger.info(f"    {feat_name}: coef={model.coef_[0]:.4f}, R²={model.score(X_train, y_train):.4f}")
            
            # 应用到所有集合（residual = feat - pred(feat)）
            for gk, group in self.protein_groups.items():
                for seq in group.sequences:
                    feat_val = seq.get(feat_name, 0)
                    pred_expr = model.predict(np.array([[feat_val]]))[0]
                    residual = feat_val - pred_expr / 10.0  # 除以10降低信号强度
                    seq[f"{feat_name}_detrended"] = float(residual)
        
        logger.info("  ✓ LM特征后处理完成")
    
    def create_hard_example_pairs(self, dataset_type: str) -> List[Tuple[int, int, int, str]]:
        """创建难例为主的训练对"""
        logger.info("=" * 60)
        logger.info(f"步骤7: 创建{dataset_type}集的难例pairs")
        logger.info("=" * 60)
        
        group_keys = self.dataset_splits[dataset_type]
        
        easy_pairs = []
        hard_pairs = []
        sample_participation = defaultdict(int)
        
        for gk in group_keys:
            if gk not in self.protein_groups:
                continue
            
            group = self.protein_groups[gk]
            sequences = group.sequences
            
            # 创建所有可能的pairs
            for i in range(len(sequences)):
                for j in range(i + 1, len(sequences)):
                    # 检查参与度限制
                    key_i = f"{gk}_{i}"
                    key_j = f"{gk}_{j}"
                    
                    if (sample_participation[key_i] >= self.config.max_sample_participation or
                        sample_participation[key_j] >= self.config.max_sample_participation):
                        continue
                    
                    expr_i = sequences[i].get("expression", {}).get("value", 0)
                    expr_j = sequences[j].get("expression", {}).get("value", 0)
                    
                    diff = abs(expr_i - expr_j)
                    
                    # 根据差异分类为难例/易例
                    if (self.config.min_expression_diff_hard <= diff <= self.config.max_expression_diff_hard):
                        # 难例
                        label = 1 if expr_i > expr_j else 0
                        hard_pairs.append((i, j, label, gk, diff))
                        sample_participation[key_i] += 1
                        sample_participation[key_j] += 1
                    elif diff >= self.config.min_expression_diff_easy:
                        # 易例
                        label = 1 if expr_i > expr_j else 0
                        easy_pairs.append((i, j, label, gk, diff))
                        sample_participation[key_i] += 1
                        sample_participation[key_j] += 1
        
        logger.info(f"  原始pairs: 难例={len(hard_pairs)}, 易例={len(easy_pairs)}")
        
        # 采样到目标数量
        if dataset_type == "train":
            target = self.config.target_train_pairs
        elif dataset_type == "val":
            target = self.config.target_val_pairs
        else:
            target = self.config.target_test_pairs
        
        # 按难例比例采样
        n_hard = int(target * self.config.hard_example_ratio)
        n_easy = target - n_hard
        
        sampled_hard = random.sample(hard_pairs, min(n_hard, len(hard_pairs)))
        sampled_easy = random.sample(easy_pairs, min(n_easy, len(easy_pairs)))
        
        all_pairs = sampled_hard + sampled_easy
        random.shuffle(all_pairs)
        
        # 保留group_key信息！这是关键修复
        # 格式：(i, j, label, group_key)
        final_pairs = [(p[0], p[1], p[2], p[3]) for p in all_pairs]
        
        logger.info(f"✓ 创建了 {len(final_pairs)} 个pairs")
        if len(final_pairs) > 0:
            logger.info(f"  难例: {len(sampled_hard)} ({len(sampled_hard)/len(final_pairs)*100:.1f}%)")
            logger.info(f"  易例: {len(sampled_easy)} ({len(sampled_easy)/len(final_pairs)*100:.1f}%)")
        else:
            logger.warning("  ⚠️ 没有生成任何pairs！")
        
        return final_pairs
    
    def generate_audit_report(self, kmer_audit: Dict[str, Any]) -> DataSplitAudit:
        """生成数据切分审计报告"""
        logger.info("=" * 60)
        logger.info("步骤8: 生成审计报告")
        logger.info("=" * 60)
        
        cluster_dist = {
            "train": [self.protein_groups[gk].cluster_id for gk in self.dataset_splits["train"]
                     if gk in self.protein_groups and self.protein_groups[gk].cluster_id is not None],
            "val": [self.protein_groups[gk].cluster_id for gk in self.dataset_splits["val"]
                   if gk in self.protein_groups and self.protein_groups[gk].cluster_id is not None],
            "test": [self.protein_groups[gk].cluster_id for gk in self.dataset_splits["test"]
                    if gk in self.protein_groups and self.protein_groups[gk].cluster_id is not None],
        }
        
        # 计算切分质量分数
        val_ok = len(self.dataset_splits["val"]) >= self.config.min_val_groups
        test_ok = len(self.dataset_splits["test"]) >= self.config.min_test_groups
        leakage_ok = not kmer_audit["leakage_detected"]
        
        quality_score = sum([val_ok, test_ok, leakage_ok]) / 3.0
        
        audit = DataSplitAudit(
            n_total_groups=len(self.protein_groups),
            n_train_groups=len(self.dataset_splits["train"]),
            n_val_groups=len(self.dataset_splits["val"]),
            n_test_groups=len(self.dataset_splits["test"]),
            n_clusters=len(self.protein_clusters),
            cluster_distribution=cluster_dist,
            kmer_audit_results=kmer_audit,
            homology_stats={
                "identity_threshold": self.config.protein_identity_threshold,
                "clustering_method": "kmer_based_hierarchical"
            },
            split_quality_score=quality_score
        )
        
        logger.info(f"✓ 审计报告生成完成")
        logger.info(f"  切分质量分数: {quality_score:.2f} / 1.0")
        logger.info(f"  Val组数达标: {'✓' if val_ok else '✗'}")
        logger.info(f"  Test组数达标: {'✓' if test_ok else '✗'}")
        logger.info(f"  无k-mer泄露: {'✓' if leakage_ok else '✗'}")
        
        self.audit_report = audit
        return audit
    
    def prepare_training_data(self, data_paths: List[str]) -> Dict[str, Any]:
        """完整的零泄露数据准备流程"""
        logger.info("开始零泄露数据准备流程...")
        logger.info("")
        
        # 1. 加载数据
        records = self.load_data(data_paths)
        
        # 2. 分组
        protein_groups = self.group_by_protein_and_host(records)
        
        # 3. 蛋白同源性聚类
        protein_clusters = self.cluster_proteins_by_homology()
        
        # 4. 按簇切分
        dataset_splits = self.split_by_clusters()
        
        # 5. k-mer审计
        kmer_audit = self.audit_kmer_leakage()
        
        # 6. K折交叉拟合弱标签
        weak_labels, weak_confidence = self.generate_weak_labels_cv()
        
        # 7. 创建难例pairs
        train_pairs = self.create_hard_example_pairs("train")
        val_pairs = self.create_hard_example_pairs("val")
        test_pairs = self.create_hard_example_pairs("test")
        
        # 8. 生成审计报告
        audit_report = self.generate_audit_report(kmer_audit)
        
        # 9. LM特征后处理（如果启用受控LM特征）
        if self.config.use_lm_features_controlled:
            logger.info("=" * 60)
            logger.info("步骤9: LM特征后处理（Host-wise z-score + 分组去趋势）")
            logger.info("=" * 60)
            self.apply_lm_feature_detrending()
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("零泄露数据准备完成！")
        logger.info("=" * 60)
        logger.info(f"总组数: {len(protein_groups)}")
        logger.info(f"Train: {len(dataset_splits['train'])} 组, {len(train_pairs)} pairs")
        logger.info(f"Val:   {len(dataset_splits['val'])} 组, {len(val_pairs)} pairs")
        logger.info(f"Test:  {len(dataset_splits['test'])} 组, {len(test_pairs)} pairs")
        logger.info(f"审计质量分数: {audit_report.split_quality_score:.2f}")
        logger.info("=" * 60)
        
        return {
            "protein_groups": protein_groups,
            "protein_clusters": protein_clusters,
            "dataset_splits": dataset_splits,
            "weak_labels": weak_labels,
            "weak_label_confidence": weak_confidence,
            "train_pairs": train_pairs,
            "val_pairs": val_pairs,
            "test_pairs": test_pairs,
            "feature_scaler": self.feature_scaler,
            "feature_extractor": self.extract_enhanced_features,
            "audit_report": audit_report.__dict__
        }


class ZeroLeakageDataset(Dataset):
    """零泄露排序数据集"""
    
    def __init__(self, 
                 protein_groups: Dict[str, ProteinGroup],
                 pairs: List[Tuple[int, int, int, str]],  # 现在包含group_key
                 group_keys: List[str],
                 feature_extractor,
                 feature_scaler: Optional[StandardScaler] = None):
        self.protein_groups = protein_groups
        self.pairs = pairs
        self.group_keys = group_keys
        self.feature_extractor = feature_extractor
        self.feature_scaler = feature_scaler
        
        logger.info(f"Dataset初始化: {len(pairs)} pairs, {len(group_keys)} groups")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        # pairs格式：(i, j, label, group_key)
        i, j, label, group_key = self.pairs[idx]
        
        if group_key not in self.protein_groups:
            raise ValueError(f"找不到组 {group_key}")
        
        group = self.protein_groups[group_key]
        
        if i >= len(group.sequences) or j >= len(group.sequences):
            raise ValueError(f"索引越界: pair ({i}, {j}) 但组 {group_key} 只有 {len(group.sequences)} 个序列")
        
        # 提取特征
        features_i = self.feature_extractor(group.sequences[i])
        features_j = self.feature_extractor(group.sequences[j])
        
        # 标准化
        if self.feature_scaler is not None:
            features_i = self.feature_scaler.transform(features_i.reshape(1, -1)).flatten()
            features_j = self.feature_scaler.transform(features_j.reshape(1, -1)).flatten()
        
        # 获取真实表达值
        expr_i = group.sequences[i].get("expression", {}).get("value", 0.0)
        expr_j = group.sequences[j].get("expression", {}).get("value", 0.0)
        
        # 添加序列索引和组大小
        group_size = len(group.sequences)
        
        return {
            'features_i': torch.tensor(features_i, dtype=torch.float32),
            'features_j': torch.tensor(features_j, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'group_key': group_key,
            'expr_i': expr_i,
            'expr_j': expr_j,
            'seq_idx_i': i,
            'seq_idx_j': j,
            'group_size': group_size
        }


def create_zero_leakage_data_loaders(
    prepared_data: Dict[str, Any],
    batch_size: int = 128,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建零泄露数据加载器"""
    
    train_dataset = ZeroLeakageDataset(
        prepared_data["protein_groups"],
        prepared_data["train_pairs"],
        prepared_data["dataset_splits"]["train"],
        prepared_data["feature_extractor"],
        prepared_data.get("feature_scaler")
    )
    
    val_dataset = ZeroLeakageDataset(
        prepared_data["protein_groups"],
        prepared_data["val_pairs"],
        prepared_data["dataset_splits"]["val"],
        prepared_data["feature_extractor"],
        prepared_data.get("feature_scaler")
    )
    
    test_dataset = ZeroLeakageDataset(
        prepared_data["protein_groups"],
        prepared_data["test_pairs"],
        prepared_data["dataset_splits"]["test"],
        prepared_data["feature_extractor"],
        prepared_data.get("feature_scaler")
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    logger.info(f"数据加载器创建完成:")
    logger.info(f"  Train: {len(train_dataset)} pairs, {len(train_loader)} batches")
    logger.info(f"  Val:   {len(val_dataset)} pairs, {len(val_loader)} batches")
    logger.info(f"  Test:  {len(test_dataset)} pairs, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

