"""
Model definition and training utilities
Contains all ML model configurations and ensemble methods
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression

from src.config import (
    USE_GPU_XGB, USE_HEAVY_MODELS,
    XGB_N_ESTIMATORS, XGB_LEARNING_RATE, XGB_MAX_DEPTH,
    RF_N_ESTIMATORS, RF_MAX_DEPTH,
    DT_MAX_DEPTH,
    LR_C, LR_PENALTY, LR_SOLVER, LR_MAX_ITER,
    RANDOM_SEED
)


def get_base_models():
    """
    Get dictionary of base machine learning models

    Returns:
        Dictionary of model_name: model_instance
    """
    models = {}

    # Logistic Regression
    models['LogReg'] = LogisticRegression(
        penalty=LR_PENALTY,
        C=LR_C,
        solver=LR_SOLVER,
        max_iter=LR_MAX_ITER,
        random_state=RANDOM_SEED
    )

    # Gaussian Naive Bayes
    models['GaussianNB'] = GaussianNB()

    # Decision Tree
    models['DecisionTree'] = DecisionTreeClassifier(
        max_depth=DT_MAX_DEPTH,
        random_state=RANDOM_SEED
    )

    # Random Forest
    models['RandomForest'] = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    # XGBoost
    xgb_params = {
        'n_estimators': XGB_N_ESTIMATORS,
        'learning_rate': XGB_LEARNING_RATE,
        'max_depth': XGB_MAX_DEPTH,
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    }

    if USE_GPU_XGB:
        try:
            # Try GPU configuration
            models['XGBoost'] = XGBClassifier(**xgb_params, device='cuda', tree_method='hist')
            print("  XGBoost: Using GPU acceleration")
        except Exception:
            # Fallback to CPU
            models['XGBoost'] = XGBClassifier(**xgb_params, tree_method='hist')
            print("  XGBoost: GPU failed, using CPU")
    else:
        models['XGBoost'] = XGBClassifier(**xgb_params, tree_method='hist')
        print("  XGBoost: Using CPU")

    # Optional heavy models (SVM, KNN)
    if USE_HEAVY_MODELS:
        models['LinearSVM'] = LinearSVC(
            C=0.1,
            max_iter=1000,
            random_state=RANDOM_SEED
        )
        models['KNN'] = KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        )
        print("  Heavy models (SVM, KNN) enabled")

    return models


def calibrate_probabilities(y_true, y_pred_proba):
    """
    Calibrate predicted probabilities using Isotonic Regression

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities

    Returns:
        Fitted IsotonicRegression calibrator
    """
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_pred_proba, y_true)
    return calibrator


def create_ensemble_predictions(base_predictions, weights=None):
    """
    Create ensemble predictions from base model predictions

    Args:
        base_predictions: Dictionary of {model_name: predictions}
        weights: Optional dictionary of {model_name: weight}

    Returns:
        Weighted average of predictions
    """
    pred_array = np.column_stack(list(base_predictions.values()))

    if weights is None:
        # Equal weighting
        return pred_array.mean(axis=1)
    else:
        # Weighted average
        weight_array = np.array([weights[name] for name in base_predictions.keys()])
        weight_array = weight_array / weight_array.sum()
        return (pred_array * weight_array).sum(axis=1)


def create_stacked_ensemble(X_train, y_train, base_predictions_train):
    """
    Create stacked ensemble using Logistic Regression

    Args:
        X_train: Original training features (not used, for API consistency)
        y_train: Training labels
        base_predictions_train: Dictionary of {model_name: predictions}

    Returns:
        Fitted LogisticRegression meta-model
    """
    # Stack base predictions as features
    X_meta = np.column_stack(list(base_predictions_train.values()))

    # Train meta-model
    meta_model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=RANDOM_SEED
    )
    meta_model.fit(X_meta, y_train)

    return meta_model


def compute_ensemble_weights(metrics_dict, metric_name='brier'):
    """
    Compute ensemble weights based on inverse metric values

    Args:
        metrics_dict: Dictionary of {model_name: metric_value}
        metric_name: Name of the metric ('brier' or 'logloss')

    Returns:
        Dictionary of {model_name: weight}
    """
    weights = {}
    for name, value in metrics_dict.items():
        # Use inverse of error metric (lower is better)
        weights[name] = 1.0 / (value + 1e-10)

    # Normalize
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    return weights
