#!/usr/bin/env python3
"""
Production Model Testing Script

Comprehensive evaluation of the trained model including:
- Performance metrics on test set
- Feature importance analysis
- Prediction distribution analysis
- Error analysis
- Conformal prediction calibration

Author: CodonVerifier Team
Date: 2025-10-05
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import joblib

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("matplotlib/seaborn not available, plotting will be skipped")

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

from codon_verifier.model_ensemble import DeepEnsemble
from scripts.train_with_all_features import EnhancedFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTester:
    """Comprehensive model testing and evaluation"""
    
    def __init__(
        self,
        model_dir: str,
        data_path: str,
        evo2_features_path: str,
        host: str = "E_coli"
    ):
        self.model_dir = Path(model_dir)
        self.data_path = data_path
        self.evo2_features_path = evo2_features_path
        self.host = host
        
        # Load models
        logger.info(f"Loading models from {model_dir}")
        self.ensemble = DeepEnsemble.load(str(self.model_dir / "ensemble.pkl"))
        self.conformal = joblib.load(str(self.model_dir / "conformal.pkl"))
        
        # Load metrics
        with open(self.model_dir / "metrics.json", 'r') as f:
            self.train_metrics = json.load(f)
        
        logger.info("✓ Models loaded successfully")
    
    def load_test_data(self, limit: int = None) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
        """Load test data"""
        logger.info(f"Loading test data from {self.data_path}")
        
        extractor = EnhancedFeatureExtractor(host=self.host)
        X, y, feature_keys, records = extractor.load_and_prepare_data(
            self.data_path,
            evo2_json=self.evo2_features_path,
            max_samples=limit
        )
        
        logger.info(f"✓ Loaded {len(y)} samples with {X.shape[1]} features")
        return X, y, feature_keys, records
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compute comprehensive metrics"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        metrics = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
            "median_ae": float(np.median(np.abs(y_true - y_pred))),
            "max_error": float(np.max(np.abs(y_true - y_pred))),
        }
        
        # Compute percentile errors
        errors = np.abs(y_true - y_pred)
        for p in [50, 75, 90, 95, 99]:
            metrics[f"error_p{p}"] = float(np.percentile(errors, p))
        
        return metrics
    
    def test_ensemble_predictions(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Test ensemble predictions"""
        logger.info("\n" + "="*60)
        logger.info("Testing Ensemble Predictions")
        logger.info("="*60)
        
        # Get predictions
        y_pred_mean, y_pred_std = self.ensemble.predict_mu_sigma(X)
        
        # Compute metrics
        metrics = self.compute_metrics(y, y_pred_mean)
        metrics["uncertainty_mean"] = float(np.mean(y_pred_std))
        metrics["uncertainty_std"] = float(np.std(y_pred_std))
        
        # Log results
        logger.info(f"R² Score: {metrics['r2']:.4f}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"MAPE: {metrics['mape']:.2f}%")
        logger.info(f"Median AE: {metrics['median_ae']:.4f}")
        logger.info(f"Max Error: {metrics['max_error']:.4f}")
        logger.info(f"Mean Uncertainty: {metrics['uncertainty_mean']:.4f} ± {metrics['uncertainty_std']:.4f}")
        
        return {
            "metrics": metrics,
            "predictions": y_pred_mean,
            "uncertainties": y_pred_std
        }
    
    def test_conformal_predictions(self, X: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> Dict:
        """Test conformal prediction intervals"""
        logger.info("\n" + "="*60)
        logger.info(f"Testing Conformal Predictions (α={alpha}, target coverage={(1-alpha)*100:.0f}%)")
        logger.info("="*60)
        
        # Get prediction intervals
        y_pred, lower, upper = self.conformal.predict_interval(X)
        # Note: predict_interval returns (mu, lower, upper)
        
        # Compute coverage
        coverage = np.mean((y >= lower) & (y <= upper))
        interval_widths = upper - lower
        
        # Compute metrics
        metrics = self.compute_metrics(y, y_pred)
        metrics["coverage"] = float(coverage)
        metrics["target_coverage"] = 1 - alpha
        metrics["coverage_error"] = float(abs(coverage - (1 - alpha)))
        metrics["interval_width_mean"] = float(np.mean(interval_widths))
        metrics["interval_width_median"] = float(np.median(interval_widths))
        metrics["interval_width_std"] = float(np.std(interval_widths))
        
        # Log results
        logger.info(f"Coverage: {coverage*100:.2f}% (target: {(1-alpha)*100:.0f}%)")
        logger.info(f"Coverage Error: {metrics['coverage_error']*100:.2f}%")
        logger.info(f"Interval Width: {metrics['interval_width_mean']:.2f} ± {metrics['interval_width_std']:.2f}")
        logger.info(f"Median Interval Width: {metrics['interval_width_median']:.2f}")
        logger.info(f"R² Score: {metrics['r2']:.4f}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        
        return {
            "metrics": metrics,
            "lower": lower,
            "upper": upper,
            "predictions": y_pred
        }
    
    def analyze_feature_importance(self, feature_keys: List[str], top_n: int = 20) -> Dict:
        """Analyze feature importance"""
        logger.info("\n" + "="*60)
        logger.info("Feature Importance Analysis")
        logger.info("="*60)
        
        # Get feature importance from each model
        importances = []
        for model in self.ensemble.models:
            # SurrogateModel has mu_model and hi_model
            # Use mu_model (median predictor) for feature importance
            if hasattr(model.mu_model, 'feature_importances_'):
                importances.append(model.mu_model.feature_importances_)
            else:
                logger.warning(f"Model {model} does not have feature_importances_")
        
        # Average importance
        avg_importance = np.mean(importances, axis=0)
        std_importance = np.std(importances, axis=0)
        
        # Sort by importance
        indices = np.argsort(avg_importance)[::-1]
        
        # Log top features
        logger.info(f"\nTop {top_n} Most Important Features:")
        for i, idx in enumerate(indices[:top_n], 1):
            logger.info(
                f"  {i:2d}. {feature_keys[idx]:40s} "
                f"{avg_importance[idx]:8.2f} ± {std_importance[idx]:6.2f}"
            )
        
        # Feature type breakdown
        feature_types = {
            "base_codon": [],
            "structural": [],
            "evolutionary": [],
            "contextual": [],
            "evo2": []
        }
        
        for idx, key in enumerate(feature_keys):
            imp = avg_importance[idx]
            if key.startswith("struct_"):
                feature_types["structural"].append(imp)
            elif key.startswith("evo_"):
                feature_types["evolutionary"].append(imp)
            elif key.startswith("ctx_"):
                feature_types["contextual"].append(imp)
            elif key.startswith("evo2_"):
                feature_types["evo2"].append(imp)
            else:
                feature_types["base_codon"].append(imp)
        
        type_importance = {
            k: {
                "mean": float(np.mean(v)) if v else 0.0,
                "sum": float(np.sum(v)) if v else 0.0,
                "count": len(v)
            }
            for k, v in feature_types.items()
        }
        
        logger.info("\nFeature Type Importance:")
        for ftype, stats in type_importance.items():
            logger.info(
                f"  {ftype:15s}: mean={stats['mean']:8.2f}, "
                f"sum={stats['sum']:10.2f}, count={stats['count']:3d}"
            )
        
        return {
            "feature_importance": {
                feature_keys[i]: float(avg_importance[i])
                for i in indices[:top_n]
            },
            "type_importance": type_importance
        }
    
    def analyze_errors(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        records: List[Dict]
    ) -> Dict:
        """Analyze prediction errors"""
        logger.info("\n" + "="*60)
        logger.info("Error Analysis")
        logger.info("="*60)
        
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        
        # Find worst predictions
        worst_indices = np.argsort(abs_errors)[-10:][::-1]
        
        logger.info("\nTop 10 Worst Predictions:")
        for i, idx in enumerate(worst_indices, 1):
            record = records[idx] if idx < len(records) else {}
            protein_id = record.get("protein_id", f"sample_{idx}")
            logger.info(
                f"  {i:2d}. {protein_id:15s} "
                f"True={y_true[idx]:6.2f}, Pred={y_pred[idx]:6.2f}, "
                f"Error={errors[idx]:+7.2f} (|{abs_errors[idx]:.2f}|)"
            )
        
        # Error distribution by expression range
        bins = [0, 30, 40, 50, 60, 100]
        bin_labels = ["<30", "30-40", "40-50", "50-60", ">60"]
        
        logger.info("\nError by Expression Range:")
        for i in range(len(bins)-1):
            mask = (y_true >= bins[i]) & (y_true < bins[i+1])
            if np.sum(mask) > 0:
                bin_mae = np.mean(abs_errors[mask])
                bin_count = np.sum(mask)
                logger.info(f"  {bin_labels[i]:8s}: MAE={bin_mae:6.2f} (n={bin_count:4d})")
        
        return {
            "worst_predictions": [
                {
                    "index": int(idx),
                    "protein_id": records[idx].get("protein_id", f"sample_{idx}") if idx < len(records) else f"sample_{idx}",
                    "true_value": float(y_true[idx]),
                    "predicted_value": float(y_pred[idx]),
                    "error": float(errors[idx]),
                    "abs_error": float(abs_errors[idx])
                }
                for idx in worst_indices
            ]
        }
    
    def generate_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainties: np.ndarray,
        output_dir: Path
    ):
        """Generate visualization plots"""
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting skipped: matplotlib/seaborn not installed")
            return
        
        logger.info("\n" + "="*60)
        logger.info("Generating Plots")
        logger.info("="*60)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Prediction vs True scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5, s=20)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('True Expression Value', fontsize=12)
        plt.ylabel('Predicted Expression Value', fontsize=12)
        plt.title('Predicted vs True Expression Values', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add R² and MAE
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.3f}',
                transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'prediction_vs_true.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: prediction_vs_true.png")
        
        # 2. Residual plot
        plt.figure(figsize=(10, 6))
        residuals = y_pred - y_true
        plt.scatter(y_pred, residuals, alpha=0.5, s=20)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted Expression Value', fontsize=12)
        plt.ylabel('Residual (Predicted - True)', fontsize=12)
        plt.title('Residual Plot', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'residual_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: residual_plot.png")
        
        # 3. Error distribution
        plt.figure(figsize=(10, 6))
        errors = np.abs(y_pred - y_true)
        plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Absolute Error', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Absolute Errors', fontsize=14, fontweight='bold')
        plt.axvline(np.mean(errors), color='r', linestyle='--', lw=2, label=f'Mean: {np.mean(errors):.2f}')
        plt.axvline(np.median(errors), color='g', linestyle='--', lw=2, label=f'Median: {np.median(errors):.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: error_distribution.png")
        
        # 4. Uncertainty vs Error
        plt.figure(figsize=(10, 6))
        plt.scatter(uncertainties, errors, alpha=0.5, s=20)
        plt.xlabel('Prediction Uncertainty (Std)', fontsize=12)
        plt.ylabel('Absolute Error', fontsize=12)
        plt.title('Uncertainty vs Absolute Error', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add correlation
        corr = np.corrcoef(uncertainties, errors)[0, 1]
        plt.text(0.05, 0.95, f'Correlation = {corr:.3f}',
                transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'uncertainty_vs_error.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: uncertainty_vs_error.png")
        
        logger.info(f"✓ All plots saved to {output_dir}")
    
    def run_comprehensive_test(self, limit: int = None, output_dir: str = None) -> Dict:
        """Run comprehensive model testing"""
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE MODEL TESTING")
        logger.info("="*60)
        logger.info(f"Model: {self.model_dir}")
        logger.info(f"Data: {self.data_path}")
        logger.info(f"Limit: {limit if limit else 'All samples'}")
        logger.info("="*60)
        
        # Load data
        X, y, feature_keys, records = self.load_test_data(limit=limit)
        
        # Test ensemble
        ensemble_results = self.test_ensemble_predictions(X, y)
        
        # Test conformal
        conformal_results = self.test_conformal_predictions(X, y)
        
        # Feature importance
        importance_results = self.analyze_feature_importance(feature_keys)
        
        # Error analysis
        error_results = self.analyze_errors(
            y, ensemble_results["predictions"], records
        )
        
        # Generate plots
        if output_dir:
            self.generate_plots(
                y,
                ensemble_results["predictions"],
                ensemble_results["uncertainties"],
                Path(output_dir)
            )
        
        # Compile results
        results = {
            "model_dir": str(self.model_dir),
            "data_path": self.data_path,
            "n_samples": len(y),
            "n_features": X.shape[1],
            "train_metrics": self.train_metrics,
            "test_results": {
                "ensemble": ensemble_results["metrics"],
                "conformal": conformal_results["metrics"],
                "feature_importance": importance_results,
                "error_analysis": error_results
            }
        }
        
        # Save results
        if output_dir:
            output_path = Path(output_dir) / "test_results.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\n✓ Results saved to {output_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Production Model Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on full dataset
  python scripts/test_production_model.py \\
    --model-dir models/production/ecoli_20251005_211800 \\
    --data data/enhanced/Ec_full_enhanced_expr.jsonl \\
    --evo2-features data/enhanced/Ec_full_evo2_features.json \\
    --output-dir test_results/full_test
  
  # Test on sample
  python scripts/test_production_model.py \\
    --model-dir models/production/ecoli_20251005_211800 \\
    --data data/enhanced/Ec_full_enhanced_expr.jsonl \\
    --evo2-features data/enhanced/Ec_full_evo2_features.json \\
    --limit 1000 \\
    --output-dir test_results/sample_test
        """
    )
    
    parser.add_argument('--model-dir', required=True,
                       help="Path to model directory")
    parser.add_argument('--data', required=True,
                       help="Path to test data JSONL")
    parser.add_argument('--evo2-features', required=True,
                       help="Path to Evo2 features JSON")
    parser.add_argument('--host', default='E_coli',
                       help="Host organism (default: E_coli)")
    parser.add_argument('--limit', type=int,
                       help="Limit number of samples to test")
    parser.add_argument('--output-dir',
                       help="Output directory for results and plots")
    
    args = parser.parse_args()
    
    try:
        # Create tester
        tester = ModelTester(
            model_dir=args.model_dir,
            data_path=args.data,
            evo2_features_path=args.evo2_features,
            host=args.host
        )
        
        # Run tests
        results = tester.run_comprehensive_test(
            limit=args.limit,
            output_dir=args.output_dir
        )
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Samples Tested: {results['n_samples']}")
        logger.info(f"Features Used: {results['n_features']}")
        logger.info(f"Test R²: {results['test_results']['ensemble']['r2']:.4f}")
        logger.info(f"Test MAE: {results['test_results']['ensemble']['mae']:.4f}")
        logger.info(f"Conformal Coverage: {results['test_results']['conformal']['coverage']*100:.2f}%")
        logger.info("="*60)
        logger.info("✅ Testing completed successfully!")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
