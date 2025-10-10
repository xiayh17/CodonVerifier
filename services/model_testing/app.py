#!/usr/bin/env python3
"""
Model Testing Service - Production Model Testing Microservice

This service handles comprehensive model testing and evaluation in the microservices architecture.
It provides detailed metrics, plots, and error analysis for trained models.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model-testing-service')

# Import testing module
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/scripts')

from test_production_model import ModelTester


def process_testing_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a model testing task"""
    start_time = time.time()
    
    try:
        config = task_data.get('config', {})
        
        logger.info("="*60)
        logger.info("MODEL TESTING SERVICE - TASK STARTED")
        logger.info("="*60)
        logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        
        # Extract configuration
        model_dir = config.get('model_dir')
        data_path = config.get('data_path')
        evo2_features_path = config.get('evo2_features_path')
        host = config.get('host', 'E_coli')
        limit = config.get('limit')
        output_dir = config.get('output_dir', '/data/test_results')
        
        # Validate required parameters
        if not model_dir:
            raise ValueError("model_dir is required")
        if not data_path:
            raise ValueError("data_path is required")
        if not evo2_features_path:
            raise ValueError("evo2_features_path is required")
        
        # Validate paths
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not Path(evo2_features_path).exists():
            raise FileNotFoundError(f"Evo2 features file not found: {evo2_features_path}")
        
        # Create tester
        logger.info(f"Initializing model tester...")
        tester = ModelTester(
            model_dir=model_dir,
            data_path=data_path,
            evo2_features_path=evo2_features_path,
            host=host
        )
        
        # Run comprehensive test
        logger.info(f"Running comprehensive model testing...")
        results = tester.run_comprehensive_test(
            limit=limit,
            output_dir=output_dir
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Prepare result
        result = {
            'status': 'success',
            'results': results,
            'output_dir': output_dir,
            'elapsed_time': elapsed_time,
            'summary': {
                'n_samples': results['n_samples'],
                'n_features': results['n_features'],
                'ensemble_r2': results['test_results']['ensemble']['r2'],
                'ensemble_mae': results['test_results']['ensemble']['mae'],
                'ensemble_rmse': results['test_results']['ensemble']['rmse'],
                'conformal_coverage': results['test_results']['conformal']['coverage'],
                'conformal_target_coverage': results['test_results']['conformal']['target_coverage']
            }
        }
        
        logger.info("="*60)
        logger.info("MODEL TESTING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Samples Tested: {result['summary']['n_samples']}")
        logger.info(f"Features Used: {result['summary']['n_features']}")
        logger.info(f"Ensemble R²: {result['summary']['ensemble_r2']:.4f}")
        logger.info(f"Ensemble MAE: {result['summary']['ensemble_mae']:.4f}")
        logger.info(f"Ensemble RMSE: {result['summary']['ensemble_rmse']:.4f}")
        logger.info(f"Conformal Coverage: {result['summary']['conformal_coverage']*100:.2f}% (target: {result['summary']['conformal_target_coverage']*100:.0f}%)")
        logger.info(f"Elapsed Time: {elapsed_time:.2f}s")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*60)
        
        return result
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Testing task failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'elapsed_time': elapsed_time
        }


def main():
    parser = argparse.ArgumentParser(
        description="Model Testing Service - Microservice for comprehensive model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Service Modes:
  1. Task Mode: Process a JSON task file
     python app.py --task-file /data/test_task.json
  
  2. Direct Mode: Run test directly with parameters
     python app.py --direct \\
       --model-dir /data/models/production/ecoli_20251007 \\
       --data /data/enhanced/Ec_full_enhanced_expr.jsonl \\
       --evo2-features /data/enhanced/Ec_full_evo2_features.json \\
       --output-dir /data/test_results \\
       --limit 1000

Task File Format:
{
  "config": {
    "model_dir": "/data/models/production/ecoli_20251007",
    "data_path": "/data/enhanced/Ec_full_enhanced_expr.jsonl",
    "evo2_features_path": "/data/enhanced/Ec_full_evo2_features.json",
    "host": "E_coli",
    "limit": 1000,
    "output_dir": "/data/test_results"
  }
}
        """
    )
    
    parser.add_argument('--task-file', type=str,
                       help="Path to JSON task file")
    parser.add_argument('--direct', action='store_true',
                       help="Run in direct mode (not from task file)")
    parser.add_argument('--model-dir', type=str,
                       help="Path to model directory (direct mode)")
    parser.add_argument('--data', type=str,
                       help="Path to test data JSONL (direct mode)")
    parser.add_argument('--evo2-features', type=str,
                       help="Path to Evo2 features JSON (direct mode)")
    parser.add_argument('--host', type=str, default='E_coli',
                       help="Host organism (direct mode)")
    parser.add_argument('--limit', type=int,
                       help="Limit number of samples (direct mode)")
    parser.add_argument('--output-dir', type=str,
                       help="Output directory for results (direct mode)")
    
    args = parser.parse_args()
    
    try:
        if args.direct:
            # Direct mode
            logger.info("Running in DIRECT mode")
            task_data = {
                'config': {
                    'model_dir': args.model_dir,
                    'data_path': args.data,
                    'evo2_features_path': args.evo2_features,
                    'host': args.host,
                    'limit': args.limit,
                    'output_dir': args.output_dir
                }
            }
        elif args.task_file:
            # Task file mode
            logger.info(f"Running in TASK FILE mode: {args.task_file}")
            with open(args.task_file, 'r') as f:
                task_data = json.load(f)
        else:
            parser.print_help()
            sys.exit(1)
        
        # Process the testing task
        result = process_testing_task(task_data)
        
        # Save result to JSON
        if result['status'] == 'success':
            output_dir = Path(result.get('output_dir', '/data/test_results'))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result_file = output_dir / 'service_result.json'
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Service result saved to: {result_file}")
            sys.exit(0)
        else:
            logger.error("Testing task failed!")
            logger.error(f"Error: {result.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Service error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
