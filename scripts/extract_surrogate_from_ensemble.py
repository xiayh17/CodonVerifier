#!/usr/bin/env python3
"""
从Ensemble模型中提取单个Surrogate模型

这个脚本从训练的DeepEnsemble中提取第一个模型，
保存为单独的surrogate.pkl文件，供序列生成使用。

用法:
    python scripts/extract_surrogate_from_ensemble.py \
        --ensemble models/production/ecoli_xxx/ensemble.pkl \
        --output models/production/ecoli_xxx/surrogate.pkl

作者: CodonVerifier Team
日期: 2025-10-08
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from codon_verifier.model_ensemble import DeepEnsemble
from codon_verifier.surrogate import SurrogateModel


def extract_surrogate(ensemble_path: str, output_path: str):
    """从ensemble中提取第一个模型作为surrogate"""
    
    print(f"加载Ensemble: {ensemble_path}")
    ensemble = DeepEnsemble.load(ensemble_path)
    
    print(f"Ensemble包含 {len(ensemble.models)} 个模型")
    
    # 提取第一个模型
    first_model = ensemble.models[0]
    
    print(f"提取第一个模型...")
    print(f"保存到: {output_path}")
    
    # 保存为surrogate
    first_model.save(output_path)
    
    print("✓ 完成！")
    print(f"\n现在您可以使用:")
    print(f"  --surrogate {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="从Ensemble提取Surrogate模型"
    )
    
    parser.add_argument('--ensemble', required=True,
                       help="Ensemble模型路径 (ensemble.pkl)")
    parser.add_argument('--output', required=True,
                       help="输出Surrogate模型路径 (surrogate.pkl)")
    
    args = parser.parse_args()
    
    # 验证输入
    if not Path(args.ensemble).exists():
        print(f"❌ Ensemble文件不存在: {args.ensemble}")
        return 1
    
    # 创建输出目录
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 提取
    try:
        extract_surrogate(args.ensemble, args.output)
        return 0
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
