"""
Model Ensemble and Uncertainty Quantification

Implements:
1. Deep Ensembles - Train multiple models for better uncertainty estimation
2. Conformal Prediction - Calibrated prediction intervals
3. Enhanced uncertainty quantification
"""

from __future__ import annotations
import numpy as np
import joblib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import warnings

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from .surrogate import SurrogateModel, SurrogateConfig


@dataclass
class EnsembleConfig:
    """Configuration for ensemble models"""
    n_models: int = 5  # Number of models in ensemble
    bootstrap: bool = True  # Bootstrap sampling for each model
    bootstrap_ratio: float = 0.9  # Ratio of data for each bootstrap
    
    # Base model config
    surrogate_config: Optional[SurrogateConfig] = None
    
    # Uncertainty estimation
    uncertainty_method: str = "ensemble_std"  # "ensemble_std" or "quantile"
    
    # Random seed
    random_state: int = 42


class DeepEnsemble:
    """
    Deep Ensemble: Train multiple surrogate models to estimate uncertainty
    
    References:
    - Lakshminarayanan et al. "Simple and Scalable Predictive Uncertainty 
      Estimation using Deep Ensembles" (NeurIPS 2017)
    """
    
    def __init__(
        self,
        feature_keys: Optional[List[str]] = None,
        cfg: Optional[EnsembleConfig] = None
    ):
        self.feature_keys = feature_keys or []
        self.cfg = cfg or EnsembleConfig()
        self.models: List[SurrogateModel] = []
        self.scaler = StandardScaler()
        
        # Initialize surrogate config if not provided
        if self.cfg.surrogate_config is None:
            self.cfg.surrogate_config = SurrogateConfig()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train ensemble of models
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        
        Returns:
            Training metrics dictionary
        """
        n_samples = len(y)
        
        # Standardize features once
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y,
            test_size=0.15,
            random_state=self.cfg.random_state
        )
        
        # Train multiple models
        self.models = []
        train_metrics = []
        
        for i in range(self.cfg.n_models):
            print(f"Training ensemble model {i+1}/{self.cfg.n_models}...")
            
            # Bootstrap sampling if enabled
            if self.cfg.bootstrap:
                n_bootstrap = int(len(X_train) * self.cfg.bootstrap_ratio)
                np.random.seed(self.cfg.random_state + i)
                indices = np.random.choice(len(X_train), size=n_bootstrap, replace=True)
                X_boot = X_train[indices]
                y_boot = y_train[indices]
            else:
                X_boot = X_train
                y_boot = y_train
            
            # Create and train model with different random seed
            config = SurrogateConfig(**self.cfg.surrogate_config.__dict__)
            config.random_state = self.cfg.random_state + i
            
            model = SurrogateModel(feature_keys=self.feature_keys, cfg=config)
            
            # Manual fit to avoid double standardization
            model.scaler = StandardScaler()
            model.scaler.fit(X_boot)  # Fit on bootstrap data
            X_boot_scaled = model.scaler.transform(X_boot)
            
            # Apply log transform if requested
            if config.use_log_transform:
                y_transformed = np.log1p(y_boot)
                model._y_is_log = True
            else:
                y_transformed = y_boot
                model._y_is_log = False
            
            # Train mu and hi models
            if _HAS_LGB:
                model.mu_model = model._make_lgb(0.5)
                model.hi_model = model._make_lgb(config.quantile_hi)
            else:
                model.mu_model = model._make_gbr(0.5)
                model.hi_model = model._make_gbr(config.quantile_hi)
            
            model.mu_model.fit(X_boot_scaled, y_transformed)
            model.hi_model.fit(X_boot_scaled, y_transformed)
            
            self.models.append(model)
        
        # Evaluate ensemble on validation set
        mu_preds, sigma_preds = self.predict_mu_sigma(X_val)
        
        r2 = r2_score(y_val, mu_preds)
        mae = mean_absolute_error(y_val, mu_preds)
        
        # Calibration: check if true values fall within predicted intervals
        lower = mu_preds - 2 * sigma_preds
        upper = mu_preds + 2 * sigma_preds
        coverage = np.mean((y_val >= lower) & (y_val <= upper))
        
        metrics = {
            "r2_ensemble": float(r2),
            "mae_ensemble": float(mae),
            "sigma_mean": float(np.mean(sigma_preds)),
            "n_val": int(len(y_val)),
            "n_models": self.cfg.n_models,
            "coverage_95": float(coverage),  # Should be ~0.95 for well-calibrated
        }
        
        return metrics
    
    def predict_mu_sigma(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with ensemble uncertainty
        
        Returns:
            mu: Mean predictions (ensemble mean)
            sigma: Uncertainty estimates (ensemble std or combined)
        """
        X_scaled = self.scaler.transform(X)
        
        # Collect predictions from all models
        all_mu = []
        all_sigma = []
        
        for model in self.models:
            X_model_scaled = model.scaler.transform(X_scaled)
            mu, sigma = model.predict_mu_sigma(X_model_scaled)
            all_mu.append(mu)
            all_sigma.append(sigma)
        
        all_mu = np.array(all_mu)  # (n_models, n_samples)
        all_sigma = np.array(all_sigma)
        
        # Ensemble mean
        mu_ensemble = np.mean(all_mu, axis=0)
        
        # Uncertainty estimation
        if self.cfg.uncertainty_method == "ensemble_std":
            # Total uncertainty = epistemic + aleatoric
            # Epistemic: variance across models
            epistemic = np.std(all_mu, axis=0)
            # Aleatoric: average of individual uncertainties
            aleatoric = np.mean(all_sigma, axis=0)
            # Total uncertainty (quadrature sum)
            sigma_ensemble = np.sqrt(epistemic**2 + aleatoric**2)
        else:  # quantile
            # Use average of individual quantile-based uncertainties
            sigma_ensemble = np.mean(all_sigma, axis=0)
        
        return mu_ensemble, sigma_ensemble
    
    def save(self, path: str):
        """Save ensemble to disk"""
        obj = {
            "feature_keys": self.feature_keys,
            "cfg": self.cfg.__dict__,
            "scaler": self.scaler,
            "models": self.models,
            "n_models": len(self.models),
        }
        joblib.dump(obj, path)
    
    @staticmethod
    def load(path: str) -> "DeepEnsemble":
        """Load ensemble from disk"""
        obj = joblib.load(path)
        
        # Reconstruct config
        cfg = EnsembleConfig(**obj["cfg"])
        
        ensemble = DeepEnsemble(feature_keys=obj["feature_keys"], cfg=cfg)
        ensemble.scaler = obj["scaler"]
        ensemble.models = obj["models"]
        
        return ensemble


@dataclass
class ConformalConfig:
    """Configuration for conformal prediction"""
    alpha: float = 0.1  # Miscoverage rate (1-alpha = coverage level)
    method: str = "quantile"  # "quantile", "cqr" (conformalized quantile regression)
    
    # For CQR (Conformalized Quantile Regression)
    use_conditional: bool = True  # Condition on features for better intervals


class ConformalPredictor:
    """
    Conformal Prediction for calibrated prediction intervals
    
    References:
    - Romano et al. "Conformalized Quantile Regression" (NeurIPS 2019)
    - Angelopoulos & Bates "A Gentle Introduction to Conformal Prediction" (2021)
    """
    
    def __init__(
        self,
        base_model: SurrogateModel,
        cfg: Optional[ConformalConfig] = None
    ):
        self.base_model = base_model
        self.cfg = cfg or ConformalConfig()
        self.calibration_scores: Optional[np.ndarray] = None
        self.q_level: float = 0.0
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Calibrate conformal predictor on calibration set
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
        """
        # Get predictions on calibration set
        mu_cal, sigma_cal = self.base_model.predict_mu_sigma(X_cal)
        
        if self.cfg.method == "quantile":
            # Simple absolute residuals
            self.calibration_scores = np.abs(y_cal - mu_cal)
        
        elif self.cfg.method == "cqr":
            # Conformalized Quantile Regression
            # Use quantile predictions to compute conformity scores
            lower_cal = mu_cal - sigma_cal
            upper_cal = mu_cal + sigma_cal
            
            # Conformity score: max(lower - y, y - upper, 0)
            self.calibration_scores = np.maximum(
                np.maximum(lower_cal - y_cal, y_cal - upper_cal),
                0.0
            )
        
        # Compute quantile for desired coverage
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.cfg.alpha)) / n
        self.q_level = float(np.quantile(self.calibration_scores, q_level))
    
    def predict_interval(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with conformal intervals
        
        Returns:
            mu: Point predictions
            lower: Lower bounds of prediction interval
            upper: Upper bounds of prediction interval
        """
        if self.calibration_scores is None:
            raise ValueError("Must calibrate before prediction. Call calibrate() first.")
        
        # Get base predictions
        mu, sigma = self.base_model.predict_mu_sigma(X)
        
        if self.cfg.method == "quantile":
            # Simple ± q_level
            lower = mu - self.q_level
            upper = mu + self.q_level
        
        elif self.cfg.method == "cqr":
            # CQR: expand quantile-based interval
            lower = mu - sigma - self.q_level
            upper = mu + sigma + self.q_level
        
        return mu, lower, upper
    
    def evaluate_coverage(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate prediction interval coverage on test set
        
        Returns:
            Metrics including empirical coverage
        """
        mu, lower, upper = self.predict_interval(X_test)
        
        # Check coverage
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        
        # Average interval width
        width = np.mean(upper - lower)
        
        # Point prediction metrics
        mae = mean_absolute_error(y_test, mu)
        r2 = r2_score(y_test, mu)
        
        return {
            "coverage": float(coverage),
            "target_coverage": float(1 - self.cfg.alpha),
            "interval_width_mean": float(width),
            "mae": float(mae),
            "r2": float(r2),
            "n_test": int(len(y_test)),
        }


def train_ensemble_with_conformal(
    X: np.ndarray,
    y: np.ndarray,
    feature_keys: List[str],
    ensemble_cfg: Optional[EnsembleConfig] = None,
    conformal_cfg: Optional[ConformalConfig] = None,
    cal_ratio: float = 0.2
) -> Tuple[DeepEnsemble, ConformalPredictor, Dict[str, Any]]:
    """
    Train ensemble model with conformal calibration
    
    Args:
        X: Feature matrix
        y: Target values
        feature_keys: Feature names
        ensemble_cfg: Ensemble configuration
        conformal_cfg: Conformal prediction configuration
        cal_ratio: Ratio of data to use for conformal calibration
    
    Returns:
        Trained ensemble, conformal predictor, and metrics
    """
    # Split data: train/cal/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train, y_train, test_size=cal_ratio / (1 - 0.15), random_state=42
    )
    
    # Train ensemble
    print(f"Training ensemble on {len(y_train)} samples...")
    ensemble = DeepEnsemble(feature_keys=feature_keys, cfg=ensemble_cfg)
    train_metrics = ensemble.fit(X_train, y_train)
    
    # Calibrate conformal predictor
    print(f"Calibrating conformal predictor on {len(y_cal)} samples...")
    # Use first model from ensemble as base for conformal
    conformal = ConformalPredictor(ensemble.models[0], cfg=conformal_cfg)
    conformal.calibrate(X_cal, y_cal)
    
    # Evaluate on test set
    print(f"Evaluating on {len(y_test)} samples...")
    conformal_metrics = conformal.evaluate_coverage(X_test, y_test)
    
    # Ensemble predictions on test set
    mu_test, sigma_test = ensemble.predict_mu_sigma(X_test)
    ensemble_test_r2 = r2_score(y_test, mu_test)
    ensemble_test_mae = mean_absolute_error(y_test, mu_test)
    
    # Combine metrics
    metrics = {
        **train_metrics,
        "conformal": conformal_metrics,
        "test_r2_ensemble": float(ensemble_test_r2),
        "test_mae_ensemble": float(ensemble_test_mae),
        "n_train": int(len(y_train)),
        "n_cal": int(len(y_cal)),
        "n_test": int(len(y_test)),
    }
    
    return ensemble, conformal, metrics

