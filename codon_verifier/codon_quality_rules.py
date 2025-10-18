"""
基于生物学规则的Codon质量评分

标签生成只用这4个基本指标：
- codon_cai
- codon_gc
- codon_rare_runs
- codon_homopolymers

模型训练可以用所有特征（包括Evo2/MSA/Structure）
"""

import numpy as np
from typing import Dict, Optional


def calculate_codon_quality_score(
    cai: Optional[float] = None,
    gc: Optional[float] = None,
    rare_runs: Optional[int] = None,
    homopolymers: Optional[int] = None,
    host: str = 'E_coli'
) -> float:
    """
    基于确定性规则计算codon质量分数
    
    规则来源：文献，不是数据拟合
    
    Returns:
        质量分数 (0-100)
    """
    score = 50.0
    
    # 1. CAI（权重最高）
    if cai is not None:
        if cai > 0.8:
            score += 25.0
        elif cai > 0.6:
            score += 15.0
        elif cai < 0.4:
            score -= 20.0
        else:
            score += 10.0 * (cai - 0.5)
    
    # 2. GC含量（最优0.45-0.55）
    if gc is not None:
        gc_dev = abs(gc - 0.50)
        if gc_dev < 0.05:
            score += 15.0
        elif gc_dev < 0.10:
            score += 8.0
        elif gc_dev > 0.20:
            score -= 20.0
        else:
            score -= 15.0 * (gc_dev - 0.10) / 0.10
    
    # 3. Rare codon runs
    if rare_runs is not None:
        if rare_runs == 0:
            score += 10.0
        else:
            score -= 6.0 * min(rare_runs, 5)
    
    # 4. Homopolymers
    if homopolymers is not None:
        if homopolymers == 0:
            score += 5.0
        else:
            score -= 4.0 * min(homopolymers, 3)
    
    return max(0.0, min(100.0, score))


def apply_quality_scores_to_records(records: list, host_key: str = 'host') -> list:
    """为所有记录添加quality score"""
    for record in records:
        host = record.get(host_key, 'E_coli')
        
        quality = calculate_codon_quality_score(
            cai=record.get('codon_cai'),
            gc=record.get('codon_gc'),
            rare_runs=record.get('codon_rare_runs'),
            homopolymers=record.get('codon_homopolymers'),
            host=host
        )
        
        # 覆盖expression字段为quality score
        record['expression'] = {
            'value': quality,
            'unit': 'codon_quality_score',
            'assay': 'rule_based',
            'confidence': 'rule_based'
        }
    
    return records

