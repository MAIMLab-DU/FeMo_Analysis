import os
import shutil
import joblib
import tarfile
import numpy as np
import pandas as pd
from typing import Literal
from sklearn.base import BaseEstimator
from ..logger import LOGGER
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
                 feature_set: Literal['crafted', 'tsfel'] = 'crafted',
                 preprocess_config: dict = None
                ) -> None:
        
        self.mu = mu
        self.dev = dev
        self.top_feat_indices = top_feat_indices
        self._feat_rank_cfg = preprocess_config.get('feature_ranking') if preprocess_config else {}
        self._feature_ranker = FeatureRanker(feature_set=feature_set, **self.feat_rank_cfg) if self.feat_rank_cfg is not None else FeatureRanker()

    def _normalize_features(
            self,
            data: np.ndarray,
            mu: np.ndarray | None = None,
            dev: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize features and remove columns with NaN values.
        
        Args:
            data: Input feature array of shape (n_samples, n_features)
            mu: Pre-computed mean values (optional)
            dev: Pre-computed deviation values (optional)
            
        Returns:
            Tuple of (normalized_features, mu, dev)
        """
        if mu is None:
            mu = np.mean(data, axis=0)
        
        # Check for NaN columns and log information
        nan_mask: np.ndarray = np.any(np.isnan(data), axis=0)
        num_nan_features: int = np.sum(nan_mask)
        
        if num_nan_features > 0:
            self.logger.warning(f"{num_nan_features} features contain NaN values and will be removed in data")
            self.logger.warning(f"NaN feature indices: {np.where(nan_mask)[0]}")
        
        norm_feats: np.ndarray = data - mu
        
        if dev is None:
            dev = np.max(norm_feats, axis=0) - np.min(norm_feats, axis=0)
        
        # Avoid division by zero by adding a small epsilon
        dev = np.where(dev == 0, 1e-8, dev)
        norm_feats = norm_feats / dev
        
        # Check for NaN columns and log information
        nan_mask: np.ndarray = np.any(np.isnan(norm_feats), axis=0)
        num_nan_features: int = np.sum(nan_mask)
        
        if num_nan_features > 0:
            self.logger.warning(f"{num_nan_features} features contain NaN values and will be removed")
            self.logger.warning(f"NaN feature indices: {np.where(nan_mask)[0]}")
        
        valid_features: np.ndarray = norm_feats[:, ~nan_mask]
        
        return valid_features, mu, dev

    def fit(self, X, y=None):
        """Fit the processor by calculating normalization parameters and ranking features.

        Args:
            X (array-like of shape (n_samples, n_features)): Input samples with n_features
            y (array-like, optional): Target values for feature ranking. Defaults to None.
        """
        
        self.logger.debug("Fitting Processor and calculating normalization parameters...")

        # Normalize features and store parameters
        X_norm, mu, dev = self._normalize_features(X, self.mu, self.dev)
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
            X (array-like of shape (n_samples, n_features)): Input samples with n_features

        Returns:
            np.ndarray: Transformed data with selected features and metadata columns.
        """
        
        self.logger.debug("Processing input data with fitted parameters...")
        if not self.is_fitted_:
            raise RuntimeError("Processor must be fitted before calling predict.")

        # Apply normalization using stored parameters
        X_norm, _, _ = self._normalize_features(X, self.mu, self.dev)
        self.logger.info(f"Normalized features shape: {X_norm.shape}")

        # Select top-ranked features
        X_norm = X_norm[:, self.top_feat_indices]

        return X_norm

    def convert_to_df(self, data: np.ndarray, columns: pd.DataFrame.columns):
        """Convert the input data to pandas Dataframe

        Args:
            data (np.ndarray): Input data array of shape (n_samples, n_features_new + 4)
            columns (list[str]): Column names of features
        """

        if self.top_feat_indices is not None:
            columns = columns[self.top_feat_indices].tolist() + ['dat_file_key', 'start_indices', 'end_indices', 'labels']
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        # Ensure proper data types
        if 'dat_file_key' in df.columns:
            df['dat_file_key'] = df['dat_file_key'].astype(str)
        
        # Convert last three columns to int
        if 'start_indices' in df.columns:
            df['start_indices'] = df['start_indices'].astype(int)
        if 'end_indices' in df.columns:
            df['end_indices'] = df['end_indices'].astype(int)
        if 'labels' in df.columns:
            df['labels'] = df['labels'].astype(int)
        
        # Ensure all feature columns are float (exclude metadata columns)
        feature_columns = [col for col in df.columns if col not in ['dat_file_key', 'start_indices', 'end_indices', 'labels']]
        for col in feature_columns:
            df[col] = df[col].astype(float)
        
        return df
    
    def save(self, file_path, key: str = 'crafted'):
        """Save the processor to a joblib file and append it to an existing tar.gz archive.

        Args:
            file_path (str): Path to directory for saving the processor.
        """
        joblib_file = os.path.join(file_path, f"{key}_processor.joblib")
        tar_file = os.path.join(file_path, "processor.tar.gz")

        # Save processor
        joblib.dump(self, joblib_file)

        # Temporary directory to extract existing tar.gz contents
        temp_dir = os.path.join(file_path, "temp_tar")
        os.makedirs(temp_dir, exist_ok=True)

        # Extract existing tar.gz contents
        if os.path.exists(tar_file):
            with tarfile.open(tar_file, "r:gz") as tar:
                tar.extractall(path=temp_dir)

        # Copy new file into temp directory
        shutil.copy(joblib_file, temp_dir)

        # Recreate tar.gz with all files
        with tarfile.open(tar_file, "w:gz") as tar:
            for filename in os.listdir(temp_dir):
                tar.add(os.path.join(temp_dir, filename), arcname=filename)

        # Cleanup temporary directory
        shutil.rmtree(temp_dir)
        self.logger.debug(f"Processor saved and appended to {tar_file}")
