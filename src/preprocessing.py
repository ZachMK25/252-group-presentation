"""
Preprocessing pipeline creation
Handles feature scaling and transformations
"""

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.data_utils import signed_log1p


def create_preprocessing_pipeline(bounded_cols, loggy_cols, the_rest):
    """
    Create preprocessing pipeline with different transformations for different feature types

    Args:
        bounded_cols: Columns that are already bounded (0-100 or 0-1) - passthrough
        loggy_cols: Columns needing log transformation then scaling
        the_rest: Columns needing only standard scaling

    Returns:
        ColumnTransformer pipeline
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('bounded', 'passthrough', bounded_cols),
            ('loggy', Pipeline([
                ('log', FunctionTransformer(signed_log1p, validate=False)),
                ('scale', StandardScaler())
            ]), loggy_cols),
            ('standard', StandardScaler(), the_rest)
        ],
        remainder='drop'
    )

    return preprocessor
