#!/usr/bin/env python3
"""
特征提取脚本 - 从整合文件中提取重要特征到单独文件

该脚本用于从整合后的JSONL文件中提取经过大量时间计算的重要特征，
包括Evo2特征和MSA特征，并保存到单独的文件中以便后续使用。

使用方法:
    python extract_features.py <input_file> [--output-dir <dir>] [--features <feature_types>]

示例:
    python extract_features.py data/enhanced/Pic_complete_v2.jsonl
    python extract_features.py data/enhanced/Pic_complete_v2.jsonl --output-dir extracted_features --features evo2,msa
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """特征提取器类"""
    
    def __init__(self, output_dir: str = "extracted_features"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 定义特征类型和对应的提取键
        self.feature_definitions = {
            'evo2': {
                'keys': [
                    'evo2_loglik', 'evo2_avg_loglik', 'evo2_perplexity', 'evo2_geom',
                    'evo2_bpb', 'evo2_delta_bpb', 'evo2_ref_bpb', 'evo2_delta_nll'
                ],
                'description': 'Evo2模型计算的特征'
            },
            'msa': {
                'keys': [
                    'msa_depth', 'msa_effective_depth', 'msa_coverage',
                    'conservation_mean', 'conservation_min', 'conservation_max', 
                    'conservation_entropy_mean', 'coevolution_score', 'contact_density',
                    'pfam_count', 'domain_count'
                ],
                'description': 'MSA (Multiple Sequence Alignment) 特征',
                'source': 'metadata.msa_features'
            },
            'evo_msa': {
                'keys': [
                    'evo_msa_depth', 'evo_msa_effective_depth', 'evo_msa_coverage',
                    'evo_conservation_mean', 'evo_conservation_min', 'evo_conservation_max',
                    'evo_conservation_entropy_mean', 'evo_coevolution_score', 'evo_contact_density',
                    'evo_pfam_count', 'evo_domain_count'
                ],
                'description': 'Evo MSA特征 (来自extra_features)'
            },
            'structural': {
                'keys': [
                    'struct_plddt_mean', 'struct_plddt_min', 'struct_plddt_max', 'struct_plddt_std',
                    'struct_plddt_q25', 'struct_plddt_q75', 'struct_disorder_ratio', 'struct_flexible_ratio',
                    'struct_sasa_mean', 'struct_sasa_total', 'struct_sasa_polar_ratio',
                    'struct_helix_ratio', 'struct_sheet_ratio', 'struct_coil_ratio',
                    'struct_has_signal_peptide', 'struct_has_transmembrane', 'struct_tm_helix_count'
                ],
                'description': '结构预测特征'
            },
            'codon': {
                'keys': [
                    'codon_gc', 'codon_cpb', 'codon_cpg_count', 'codon_cpg_freq',
                    'codon_cpg_obs_exp', 'codon_upa_count', 'codon_upa_freq', 'codon_upa_obs_exp',
                    'codon_rare_runs', 'codon_rare_run_total_len', 'codon_homopolymers', 'codon_homopoly_total_len'
                ],
                'description': '密码子使用特征'
            },
            'context': {
                'keys': [
                    'ctx_promoter_strength', 'ctx_rbs_strength', 'ctx_rbs_spacing', 'ctx_kozak_score',
                    'ctx_vector_copy_number', 'ctx_has_selection_marker', 'ctx_temperature_norm',
                    'ctx_inducer_concentration', 'ctx_growth_phase_score', 'ctx_localization_score'
                ],
                'description': '表达上下文特征'
            }
        }
    
    def extract_features_from_record(self, record: Dict[str, Any], feature_types: List[str]) -> Dict[str, Dict[str, Any]]:
        """从单条记录中提取指定类型的特征"""
        extracted = {}
        
        # 基本信息
        basic_info = {
            'sequence': record.get('sequence', ''),
            'protein_aa': record.get('protein_aa', ''),
            'host': record.get('host', ''),
            'protein_id': record.get('protein_id', ''),
            'uniprot_id': record.get('metadata', {}).get('uniprot_id', ''),
            'entry_name': record.get('metadata', {}).get('entry_name', ''),
            'organism': record.get('metadata', {}).get('organism', '')
        }
        
        # 获取extra_features和msa_features
        extra_features = record.get('extra_features', {})
        msa_features = record.get('metadata', {}).get('msa_features', {})
        
        for feature_type in feature_types:
            if feature_type not in self.feature_definitions:
                logger.warning(f"未知的特征类型: {feature_type}")
                continue
                
            feature_keys = self.feature_definitions[feature_type]['keys']
            feature_data = {}
            
            # 提取特征值
            for key in feature_keys:
                if key in extra_features:
                    feature_data[key] = extra_features[key]
                elif key in msa_features:
                    feature_data[key] = msa_features[key]
                else:
                    # 如果找不到，记录为None
                    feature_data[key] = None
                    logger.debug(f"未找到特征 {key} 在记录中")
            
            # 合并基本信息和特征数据
            extracted[feature_type] = {**basic_info, **feature_data}
        
        return extracted
    
    def extract_from_file(self, input_file: str, feature_types: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """从文件中提取特征"""
        logger.info(f"开始从文件 {input_file} 提取特征: {', '.join(feature_types)}")
        
        all_extracted = {ft: [] for ft in feature_types}
        total_records = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    extracted = self.extract_features_from_record(record, feature_types)
                    
                    for feature_type, data in extracted.items():
                        all_extracted[feature_type].append(data)
                    
                    total_records += 1
                    
                    if line_num % 1000 == 0:
                        logger.info(f"已处理 {line_num} 条记录")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"第 {line_num} 行JSON解析错误: {e}")
                    continue
                except Exception as e:
                    logger.error(f"第 {line_num} 行处理错误: {e}")
                    continue
        
        logger.info(f"总共处理了 {total_records} 条记录")
        return all_extracted
    
    def save_features(self, extracted_features: Dict[str, List[Dict[str, Any]]], input_file: str):
        """保存提取的特征到文件"""
        input_path = Path(input_file)
        base_name = input_path.stem
        
        for feature_type, records in extracted_features.items():
            if not records:
                logger.warning(f"没有提取到 {feature_type} 特征")
                continue
            
            # 生成输出文件名
            output_file = self.output_dir / f"{base_name}_{feature_type}_features.jsonl"
            
            # 保存特征
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            logger.info(f"保存了 {len(records)} 条 {feature_type} 特征到 {output_file}")
            
            # 生成统计信息
            self._generate_stats(feature_type, records, output_file)
    
    def _generate_stats(self, feature_type: str, records: List[Dict[str, Any]], output_file: Path):
        """生成特征统计信息"""
        if not records:
            return
        
        stats = {
            'feature_type': feature_type,
            'total_records': len(records),
            'feature_description': self.feature_definitions[feature_type]['description'],
            'available_features': [],
            'missing_features': [],
            'feature_stats': {}
        }
        
        # 分析特征可用性
        feature_keys = self.feature_definitions[feature_type]['keys']
        for key in feature_keys:
            non_null_count = sum(1 for record in records if record.get(key) is not None)
            if non_null_count > 0:
                stats['available_features'].append(key)
                stats['feature_stats'][key] = {
                    'available_count': non_null_count,
                    'availability_rate': non_null_count / len(records)
                }
            else:
                stats['missing_features'].append(key)
        
        # 保存统计信息
        stats_file = output_file.with_suffix('.stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"生成统计信息: {stats_file}")
        logger.info(f"  - 可用特征: {len(stats['available_features'])}/{len(feature_keys)}")
        logger.info(f"  - 缺失特征: {len(stats['missing_features'])}")

def main():
    parser = argparse.ArgumentParser(description='从整合文件中提取重要特征')
    parser.add_argument('input_file', help='输入的JSONL文件路径')
    parser.add_argument('--output-dir', default='extracted_features', 
                       help='输出目录 (默认: extracted_features)')
    parser.add_argument('--features', default='evo2,msa,evo_msa,structural,codon,context',
                       help='要提取的特征类型，用逗号分隔 (默认: 所有类型)')
    parser.add_argument('--list-features', action='store_true',
                       help='列出所有可用的特征类型')
    
    args = parser.parse_args()
    
    # 列出可用特征类型
    if args.list_features:
        extractor = FeatureExtractor()
        print("可用的特征类型:")
        for feature_type, info in extractor.feature_definitions.items():
            print(f"  {feature_type}: {info['description']}")
            print(f"    包含特征: {', '.join(info['keys'])}")
        return
    
    # 检查输入文件
    if not os.path.exists(args.input_file):
        logger.error(f"输入文件不存在: {args.input_file}")
        return
    
    # 解析特征类型
    feature_types = [ft.strip() for ft in args.features.split(',')]
    
    # 创建提取器并执行提取
    extractor = FeatureExtractor(args.output_dir)
    extracted_features = extractor.extract_from_file(args.input_file, feature_types)
    extractor.save_features(extracted_features, args.input_file)
    
    logger.info("特征提取完成!")

if __name__ == '__main__':
    main()
