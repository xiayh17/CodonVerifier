#!/usr/bin/env python3
"""
弱标签数据分析脚本
分析五个物种的弱标签数据并生成检查报告
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class WeakLabelsAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.species_files = {
            'E_coli': 'Ec_complete_v3_updated.jsonl',
            'Human': 'Human_complete_v3_updated.jsonl', 
            'Mouse': 'mouse_complete_v3_updated.jsonl',
            'P_pastoris': 'Pic_complete_v3_updated.jsonl',
            'S_cerevisiae': 'Sac_complete_v3_updated.jsonl'
        }
        self.data = {}
        self.combined_data = None
        
    def load_data(self):
        """加载所有物种的数据"""
        print("正在加载数据...")
        
        for species, filename in self.species_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                print(f"加载 {species} 数据...")
                data_list = []
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data_list.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
                
                df = pd.DataFrame(data_list)
                df['species'] = species
                
                # 展开嵌套的字典结构
                if 'msa_features' in df.columns:
                    msa_df = pd.json_normalize(df['msa_features'])
                    msa_df.columns = [f'msa_{col}' for col in msa_df.columns]
                    df = pd.concat([df.drop('msa_features', axis=1), msa_df], axis=1)
                
                if 'expression' in df.columns:
                    expr_df = pd.json_normalize(df['expression'])
                    expr_df.columns = [f'expr_{col}' for col in expr_df.columns]
                    df = pd.concat([df.drop('expression', axis=1), expr_df], axis=1)
                
                self.data[species] = df
                print(f"  - {species}: {len(df)} 条记录")
            else:
                print(f"警告: 文件 {filepath} 不存在")
        
        # 合并所有数据
        if self.data:
            self.combined_data = pd.concat(self.data.values(), ignore_index=True)
            print(f"\n总共加载了 {len(self.combined_data)} 条记录")
        
    def basic_statistics(self):
        """基本统计信息"""
        print("\n=== 基本统计信息 ===")
        
        # 物种分布
        species_counts = self.combined_data['species'].value_counts()
        print("\n物种分布:")
        for species, count in species_counts.items():
            print(f"  {species}: {count} 条记录")
        
        # 序列长度统计
        self.combined_data['seq_length'] = self.combined_data['sequence'].str.len()
        self.combined_data['aa_length'] = self.combined_data['protein_aa'].str.len()
        
        print(f"\n序列长度统计:")
        print(f"  DNA序列长度: {self.combined_data['seq_length'].mean():.1f} ± {self.combined_data['seq_length'].std():.1f}")
        print(f"  蛋白质长度: {self.combined_data['aa_length'].mean():.1f} ± {self.combined_data['aa_length'].std():.1f}")
        
        # 表达值统计
        if 'expr_value' in self.combined_data.columns:
            expression_values = self.combined_data['expr_value']
        elif 'expression_value' in self.combined_data.columns:
            expression_values = self.combined_data['expression_value']
        else:
            expression_values = self.combined_data['expression'].apply(lambda x: x['value'] if isinstance(x, dict) else x)
        
        print(f"\n表达值统计:")
        print(f"  平均值: {expression_values.mean():.1f}")
        print(f"  中位数: {expression_values.median():.1f}")
        print(f"  标准差: {expression_values.std():.1f}")
        
        return {
            'species_counts': species_counts,
            'seq_length_stats': self.combined_data['seq_length'].describe(),
            'aa_length_stats': self.combined_data['aa_length'].describe(),
            'expression_stats': expression_values.describe()
        }
    
    def extract_features(self):
        """提取关键特征"""
        print("\n=== 特征提取 ===")
        
        # 提取MSA特征
        msa_features = ['msa_msa_depth', 'msa_msa_effective_depth', 'msa_msa_coverage', 
                       'msa_conservation_mean', 'msa_conservation_min', 'msa_conservation_max']
        
        # 提取Evo2特征
        evo2_features = ['evo2_loglik', 'evo2_avg_loglik', 'evo2_perplexity', 
                        'evo2_geom', 'evo2_bpb']
        
        # 提取密码子特征
        codon_features = ['codon_gc', 'codon_cpg_freq', 'codon_upa_freq', 
                         'codon_rare_runs', 'codon_homopolymers']
        
        # 提取结构特征
        struct_features = ['struct_plddt_mean', 'struct_plddt_min', 'struct_plddt_max',
                          'struct_disorder_ratio', 'struct_flexible_ratio',
                          'struct_helix_ratio', 'struct_sheet_ratio', 'struct_coil_ratio']
        
        # 提取表达特征
        if 'expr_value' in self.combined_data.columns:
            self.combined_data['expression_value'] = self.combined_data['expr_value']
        else:
            expression_values = self.combined_data['expression'].apply(
                lambda x: x['value'] if isinstance(x, dict) else x
            )
            self.combined_data['expression_value'] = expression_values
        
        all_features = msa_features + evo2_features + codon_features + struct_features + ['expression_value']
        
        # 检查特征可用性
        available_features = []
        missing_features = []
        
        for feature in all_features:
            if feature in self.combined_data.columns:
                available_features.append(feature)
            else:
                missing_features.append(feature)
        
        print(f"可用特征: {len(available_features)} 个")
        print(f"缺失特征: {len(missing_features)} 个")
        if missing_features:
            print(f"缺失的特征: {missing_features}")
        
        return available_features, missing_features
    
    def create_visualizations(self, output_dir):
        """创建可视化图表"""
        print("\n=== 创建可视化图表 ===")
        
        # 确保序列长度列存在
        if 'seq_length' not in self.combined_data.columns:
            self.combined_data['seq_length'] = self.combined_data['sequence'].str.len()
        if 'aa_length' not in self.combined_data.columns:
            self.combined_data['aa_length'] = self.combined_data['protein_aa'].str.len()
        if 'expression_value' not in self.combined_data.columns:
            if 'expr_value' in self.combined_data.columns:
                self.combined_data['expression_value'] = self.combined_data['expr_value']
            else:
                # 如果expression列仍然存在，使用原始方法
                if 'expression' in self.combined_data.columns:
                    expression_values = self.combined_data['expression'].apply(
                        lambda x: x['value'] if isinstance(x, dict) else x
                    )
                    self.combined_data['expression_value'] = expression_values
                else:
                    # 如果没有expression相关列，创建一个默认值
                    self.combined_data['expression_value'] = 50.0
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. 物种分布饼图
        plt.figure(figsize=(10, 8))
        species_counts = self.combined_data['species'].value_counts()
        plt.pie(species_counts.values, labels=species_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('物种分布', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'species_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 序列长度分布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # DNA序列长度
        for species in self.combined_data['species'].unique():
            species_data = self.combined_data[self.combined_data['species'] == species]
            ax1.hist(species_data['seq_length'], alpha=0.7, label=species, bins=30)
        ax1.set_xlabel('DNA序列长度 (bp)')
        ax1.set_ylabel('频次')
        ax1.set_title('DNA序列长度分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 蛋白质长度
        for species in self.combined_data['species'].unique():
            species_data = self.combined_data[self.combined_data['species'] == species]
            ax2.hist(species_data['aa_length'], alpha=0.7, label=species, bins=30)
        ax2.set_xlabel('蛋白质长度 (氨基酸)')
        ax2.set_ylabel('频次')
        ax2.set_title('蛋白质长度分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'sequence_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 表达值分布
        plt.figure(figsize=(12, 8))
        for i, species in enumerate(self.combined_data['species'].unique(), 1):
            plt.subplot(2, 3, i)
            species_data = self.combined_data[self.combined_data['species'] == species]
            expression_values = species_data['expression_value']
            plt.hist(expression_values, bins=30, alpha=0.7, color=f'C{i-1}')
            plt.title(f'{species}\n表达值分布')
            plt.xlabel('表达值')
            plt.ylabel('频次')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'expression_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 关键特征箱线图
        key_features = ['msa_msa_depth', 'msa_conservation_mean', 'evo2_perplexity', 
                       'codon_gc', 'struct_plddt_mean', 'expression_value']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(key_features):
            if feature in self.combined_data.columns:
                sns.boxplot(data=self.combined_data, x='species', y=feature, ax=axes[i])
                axes[i].set_title(f'{feature} 分布')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'key_features_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 特征相关性热图
        numeric_features = ['seq_length', 'aa_length', 'msa_msa_depth', 'msa_conservation_mean', 
                           'evo2_perplexity', 'codon_gc', 'struct_plddt_mean', 'expression_value']
        
        available_numeric = [f for f in numeric_features if f in self.combined_data.columns]
        
        if len(available_numeric) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.combined_data[available_numeric].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f')
            plt.title('特征相关性热图', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path / 'feature_correlation.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"可视化图表已保存到: {output_path}")
    
    def quality_assessment(self):
        """数据质量评估"""
        print("\n=== 数据质量评估 ===")
        
        quality_report = {}
        
        # 1. 缺失值检查
        missing_data = self.combined_data.isnull().sum()
        missing_percentage = (missing_data / len(self.combined_data)) * 100
        
        print("缺失值统计:")
        high_missing = missing_percentage[missing_percentage > 10]
        if len(high_missing) > 0:
            print("高缺失率特征 (>10%):")
            for feature, percentage in high_missing.items():
                print(f"  {feature}: {percentage:.1f}%")
        else:
            print("  所有特征缺失率都低于10%")
        
        quality_report['missing_data'] = missing_percentage.to_dict()
        
        # 2. 异常值检查
        numeric_cols = self.combined_data.select_dtypes(include=[np.number]).columns
        outliers_report = {}
        
        for col in numeric_cols:
            if col in self.combined_data.columns:
                Q1 = self.combined_data[col].quantile(0.25)
                Q3 = self.combined_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.combined_data[(self.combined_data[col] < lower_bound) | 
                                            (self.combined_data[col] > upper_bound)]
                outlier_percentage = (len(outliers) / len(self.combined_data)) * 100
                
                if outlier_percentage > 5:  # 只报告异常值比例超过5%的特征
                    outliers_report[col] = outlier_percentage
        
        if outliers_report:
            print("\n异常值统计 (>5%):")
            for feature, percentage in outliers_report.items():
                print(f"  {feature}: {percentage:.1f}%")
        else:
            print("\n所有特征的异常值比例都低于5%")
        
        quality_report['outliers'] = outliers_report
        
        # 3. 数据一致性检查
        print("\n数据一致性检查:")
        
        # 检查序列长度与蛋白质长度的关系
        self.combined_data['expected_aa_length'] = self.combined_data['seq_length'] / 3
        length_consistency = abs(self.combined_data['aa_length'] - self.combined_data['expected_aa_length']) <= 1
        consistency_rate = length_consistency.mean() * 100
        
        print(f"  序列长度一致性: {consistency_rate:.1f}%")
        quality_report['length_consistency'] = consistency_rate
        
        # 检查表达值的合理性
        expression_values = self.combined_data['expression_value']
        valid_expression = (expression_values >= 0) & (expression_values <= 100)
        valid_expression_rate = valid_expression.mean() * 100
        
        print(f"  表达值合理性: {valid_expression_rate:.1f}%")
        quality_report['expression_validity'] = valid_expression_rate
        
        return quality_report
    
    def generate_report(self, output_dir):
        """生成综合报告"""
        print("\n=== 生成综合报告 ===")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 基本统计
        basic_stats = self.basic_statistics()
        
        # 特征提取
        available_features, missing_features = self.extract_features()
        
        # 质量评估
        quality_report = self.quality_assessment()
        
        # 生成报告
        report_content = f"""
# 弱标签数据检查报告

## 数据概览

### 物种分布
"""
        for species, count in basic_stats['species_counts'].items():
            report_content += f"- **{species}**: {count} 条记录\n"
        
        report_content += f"""
### 数据规模
- **总记录数**: {len(self.combined_data):,}
- **平均DNA序列长度**: {basic_stats['seq_length_stats']['mean']:.1f} bp
- **平均蛋白质长度**: {basic_stats['aa_length_stats']['mean']:.1f} 氨基酸
- **平均表达值**: {basic_stats['expression_stats']['mean']:.1f}

## 特征分析

### 可用特征 ({len(available_features)} 个)
- **MSA特征**: msa_depth, msa_effective_depth, msa_coverage, conservation_mean等
- **Evo2特征**: evo2_loglik, evo2_perplexity, evo2_bpb等  
- **密码子特征**: codon_gc, codon_cpg_freq, codon_upa_freq等
- **结构特征**: struct_plddt_mean, struct_helix_ratio, struct_sheet_ratio等
- **表达特征**: expression_value

### 缺失特征 ({len(missing_features)} 个)
"""
        if missing_features:
            for feature in missing_features:
                report_content += f"- {feature}\n"
        else:
            report_content += "- 无缺失特征\n"
        
        report_content += f"""
## 数据质量评估

### 缺失值分析
"""
        high_missing = {k: v for k, v in quality_report['missing_data'].items() if v > 10}
        if high_missing:
            report_content += "高缺失率特征 (>10%):\n"
            for feature, percentage in high_missing.items():
                report_content += f"- **{feature}**: {percentage:.1f}%\n"
        else:
            report_content += "- 所有特征缺失率都低于10%\n"
        
        report_content += f"""
### 异常值分析
"""
        if quality_report['outliers']:
            report_content += "高异常值比例特征 (>5%):\n"
            for feature, percentage in quality_report['outliers'].items():
                report_content += f"- **{feature}**: {percentage:.1f}%\n"
        else:
            report_content += "- 所有特征的异常值比例都低于5%\n"
        
        report_content += f"""
### 数据一致性
- **序列长度一致性**: {quality_report['length_consistency']:.1f}%
- **表达值合理性**: {quality_report['expression_validity']:.1f}%

## 物种特异性分析

### 各物种特征统计
"""
        
        # 为每个物种生成统计信息
        for species in self.combined_data['species'].unique():
            species_data = self.combined_data[self.combined_data['species'] == species]
            report_content += f"""
#### {species}
- **记录数**: {len(species_data):,}
- **平均序列长度**: {species_data['seq_length'].mean():.1f} bp
- **平均蛋白质长度**: {species_data['aa_length'].mean():.1f} 氨基酸
- **平均表达值**: {species_data['expression_value'].mean():.1f}
- **平均MSA深度**: {species_data['msa_msa_depth'].mean():.1f}
- **平均保守性**: {species_data['msa_conservation_mean'].mean():.3f}
- **平均结构置信度**: {species_data['struct_plddt_mean'].mean():.1f}
"""
        
        report_content += f"""
## 建议

### 数据质量改进建议
1. **缺失值处理**: 对于缺失率较高的特征，建议进行插补或删除
2. **异常值处理**: 对于异常值比例较高的特征，建议进行异常值检测和处理
3. **数据验证**: 建议对序列长度和蛋白质长度的一致性进行进一步验证

### 分析建议
1. **特征选择**: 基于相关性分析选择最重要的特征进行建模
2. **物种特异性**: 考虑不同物种的特征差异，可能需要分别建模
3. **表达预测**: 可以基于现有特征构建表达预测模型

## 可视化图表
- 物种分布图: species_distribution.png
- 序列长度分布图: sequence_length_distribution.png  
- 表达值分布图: expression_distribution.png
- 关键特征箱线图: key_features_boxplot.png
- 特征相关性热图: feature_correlation.png

---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 保存报告
        with open(output_path / 'weaklabels_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"报告已保存到: {output_path / 'weaklabels_report.md'}")
        
        return report_content

def main():
    """主函数"""
    # 数据目录
    data_dir = "/mnt/c/Users/xiayh17/Documents/GitHub/CodonVerifier/data/weaklabels"
    output_dir = "/mnt/c/Users/xiayh17/Documents/GitHub/CodonVerifier/reports/weaklabels_analysis"
    
    # 创建分析器
    analyzer = WeakLabelsAnalyzer(data_dir)
    
    # 加载数据
    analyzer.load_data()
    
    if analyzer.combined_data is not None:
        # 创建可视化
        analyzer.create_visualizations(output_dir)
        
        # 生成报告
        analyzer.generate_report(output_dir)
        
        print(f"\n分析完成！结果保存在: {output_dir}")
    else:
        print("错误: 无法加载数据")

if __name__ == "__main__":
    main()
