import os
import joblib
import tarfile
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from ..logger import LOGGER
from .utils import normalize_features
from .ranking import FeatureRanker

class Processor(BaseEstimator):
    @property
    def logger(self):
        return LOGGER
    
    @property
    def feat_rank_cfg(self) -> dict:
        return self._feat_rank_cfg
    
    def __init__(self,
                 *,
                 mu: np.ndarray = None,
                 dev: np.ndarray = None,
                 top_feat_indices: np.ndarray = None,
                 preprocess_config: dict = None
                ) -> None:
        
        self.mu = mu
        self.dev = dev
        self.top_feat_indices = top_feat_indices
        self._feat_rank_cfg = preprocess_config.get('feature_ranking') if preprocess_config else {}
        self._feature_ranker = FeatureRanker(**self.feat_rank_cfg) if self.feat_rank_cfg is not None else FeatureRanker()

    def fit(self, X, y=None):
        """Fit the processor by calculating normalization parameters and ranking features.

        Args:
            X (array-like of shape (n_samples, n_features + 3)): Input samples with n_features + ['filename_hash', 'det_indices', 'labels']
            y (array-like, optional): Target values for feature ranking. Defaults to None.
        """
        
        self.logger.debug("Fitting Processor and calculating normalization parameters...")

        # Separate additional metadata columns and store normalization parameters
        X = X[:, :-2]  # Exclude metadata columns for feature normalization

        # Normalize features and store parameters
        X_norm, mu, dev = normalize_features(X, self.mu, self.dev)
        self.mu, self.dev = mu, dev
        
        # Rank features and store the top feature indices
        top_feat_indices = self._feature_ranker.fit(X_norm, y, func=self._feature_ranker.ensemble_ranking)
        self.top_feat_indices = top_feat_indices
        
        self.is_fitted_ = True
        self.logger.debug("Processor fitted with normalization and feature ranking.")
        return self

    def predict(self, X):
        """Process the input data by applying normalization and feature selection, based on fitted parameters.

        Args:
            X (array-like of shape (n_samples, n_features + 2)): Input samples with n_features + ['filename_hash', 'det_indices']

        Returns:
            np.ndarray: Transformed data with selected features and metadata columns.
        """
        
        self.logger.debug("Processing input data with fitted parameters...")
        if not self.is_fitted_:
            raise RuntimeError("Processor must be fitted before calling predict.")

        det_indices = X[:, -1]
        filename_hash = X[:, -2]
        X = X[:, :-2]  # Exclude metadata columns for feature normalization

        # Apply normalization using stored parameters
        X_norm, _, _ = normalize_features(X, self.mu, self.dev)

        # Select top-ranked features
        X_norm = X_norm[:, self.top_feat_indices]

        # Concatenate metadata columns and return
        processed_data = np.concatenate(
            [X_norm, filename_hash[:, np.newaxis], det_indices[:, np.newaxis]],
            axis=1
        )
        return processed_data

    def convert_to_df(self, data: np.ndarray, columns: pd.DataFrame.columns):
        """Convert the input data to pandas Dataframe

        Args:
            data (np.ndarray): Input data array of shape (n_samples, n_features_new + 3)
            columns (list[str]): Column names of features
        """

        if self.top_feat_indices is not None:
            columns = columns[self.top_feat_indices].tolist() + ['filename_hash', 'det_indices', 'labels']
        return pd.DataFrame(data, columns=columns)
    
    def save(self, file_path):
        """Save the processor to a joblib file

        Args:
            file_path (str): Path to directory for saving the processor
        """
        
        joblib.dump(self, os.path.join(file_path, "processor.joblib"))
        tar = tarfile.open(os.path.join(file_path, "processor.tar.gz"), "w:gz")
        tar.add(os.path.join(file_path, "processor.joblib"), arcname="processor.joblib")
        tar.close()
        self.logger.debug(f"Processor saved to {file_path}")
