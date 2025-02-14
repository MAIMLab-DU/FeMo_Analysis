"""
@author: Monaf Chowdhury
This module contains FeatureRanker class for feature ranking using various methods
such as PCA, NCA, RFE, SelectKBest, and XGBoost.
For some guidelines, check out [this page](https://scikit-learn.org/stable/modules/feature_selection.html)
"""

import itertools
import numpy as np
from typing import Literal
from ..logger import LOGGER
from functools import wraps
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier


class FeatureRanker:

    @property
    def logger(self):
        return LOGGER

    def __init__(self,
                 feature_set: Literal['crafted', 'tsfel'] = 'crafted',
                 feat_ratio: int = 3,
                 param_cfg: dict = {
                     'nca_ranking': {
                         'n_components': 30,
                         'max_iter': 1000
                     },
                     'xgb_ranking': {
                         'learning_rate': 0.01,
                         'n_estimators': 100,
                         'max_depth': 3
                     },
                     'logReg_ranking': {
                         'solver': 'liblinear'
                     },
                     'recursive_ranking': {
                         'step': 1,
                         'estimator': 'AdaBoost'
                     }
                 }
                 ) -> None:
        """Initializes the FeatureRanker class

        Args:
            min_method (int, optional): How many feature selection methods must have common features. Defaults to 2.
            feat_ratio (int, optional): Taking only the 1/n-th of the main feature space. Defaults to 3.
            param_cfg (dict, optional): Configuration parameters for different feature selection methods. Defaults to {
                'nca_ranker': {'n_components': 30,'max_iter': 1000},
                'xgb_ranker': {'learning_rate': 0.01, 'n_estimators': 100,'max_depth': 3},
                'logReg_ranker': {'solver': 'liblinear'},
                'recursive_ranker: {'step': 1}
        """

        self._min_common = 2 if feature_set == 'crafted' else 3
        self._feature_ratio = feat_ratio
        self._param_cfg = param_cfg

    def _nca_ranking(self, X, y):
        """"Ranks features based on their importance using Neighboord Component Analysis.

        Args:
            X (array-like of shape (n_samples, n_features)): The training samples.
            y (array-like of shape (n_samples,)): The corresponding training labels.

        Returns:
            np.ndarray:
             n-most important feature indices
        """
        self.logger.debug("Neighbourhood Component Analysis is going on...\n")
        n_top_feats = X.shape[1] // self._feature_ratio
        nca = NeighborhoodComponentsAnalysis(init='auto',
                                             tol=1e-5,
                                             verbose=1,
                                             random_state=0,
                                             **self._param_cfg.get('nca_ranking'))
        nca.fit(X, y)
        nca_top_indices = np.argsort(np.abs(nca.components_))[0][-n_top_feats:][::-1]
        self.logger.debug(f"NCA top {len(nca_top_indices)} features: {nca_top_indices}\n")
        return nca_top_indices
    
    def _xgboost_ranking(self, X, y):
        """"Ranks features based on their importance using an XGBoost classifier.

        Args:
            X (array-like of shape (n_samples, n_features)): The training samples.
            y (array-like of shape (n_samples,)): The corresponding training labels.

        Returns:
            np.ndarray:
             n-most important feature indices
        """
        self.logger.debug("XGBoost Ranking is going on... \n")
        n_top_feats = X.shape[1] // self._feature_ratio
        xgb = XGBClassifier(random_state=0,
                            booster = 'gbtree',
                            **self._param_cfg.get('xgb_ranking'))
        xgb.fit(X, y)
        feature_importances = xgb.feature_importances_
        xgb_top_indices = np.argsort(feature_importances)[-n_top_feats:][::-1]
        self.logger.debug(f"XGBoost top {len(xgb_top_indices)} features: {xgb_top_indices}\n")
        return xgb_top_indices

    def _logReg_ranking(self, X, y):
        """"Ranks features based on their importance using a Logistic Regressor.

        Args:
            X (array-like of shape (n_samples, n_features)): The training samples.
            y (array-like of shape (n_samples,)): The corresponding training labels.

        Returns:
            np.ndarray:
             n-most important feature indices
        """
        self.logger.debug("L1 based feature selection is going on... \n")
        
        log_reg = LogisticRegression(penalty='l1',
                                     random_state=42,
                                     **self._param_cfg.get('logReg_ranking'))
        log_reg.fit(X, y)
        l1_top_indices = SelectFromModel(log_reg, prefit=True).get_support(indices=True)
        self.logger.debug(f"L1 based top {len(l1_top_indices)} features: {l1_top_indices}\n")
        
        return l1_top_indices
    
    def _recursive_ranking(self, X, y):
        """"Ranks features based on their importance using Recursive Feature Elimination.

        Args:
            X (array-like of shape (n_samples, n_features)): The training samples.
            y (array-like of shape (n_samples,)): The corresponding training labels.

        Returns:
            np.ndarray:
             n-most important feature indices
        """
        self.logger.debug("Recursive feature elimination is going on... \n")
        n_top_feats = X.shape[1] // self._feature_ratio
        estimator_type = self._param_cfg.get('recursive_ranking').get('estimator', 'AdaBoost')
        
        if estimator_type == 'AdaBoost':
            estimator = AdaBoostClassifier(random_state=0)
        elif estimator_type == 'ExtraTrees':
            estimator = ExtraTreesClassifier(random_state=0)  # gives slightly low f1 score.
        else:
            raise ValueError(f"Invalid {estimator = }")
        kwargs = {k: v for k, v in self._param_cfg.get('recursive_ranking').items() if k != 'estimator'}
        rfe = RFE(estimator=estimator,
                  n_features_to_select=n_top_feats,
                  verbose=1,
                  **kwargs)
        rfe.fit(X, y)
        recursive_top_indices = rfe.get_support(indices=True)
        
        self.logger.debug(f"RFE top {len(recursive_top_indices)} features: {recursive_top_indices}\n")
        
        return recursive_top_indices
    
    def ensemble_ranking(self, X, y):
        """"Ranks features based on their importance using Recursive Feature Elimination.

        Args:
            X (array-like of shape (n_samples, n_features)): The training samples.
            y (array-like of shape (n_samples,)): The corresponding training labels.

        Returns:
            np.ndarray: 
             n-most important feature indices
        """
        nca_top_n = self._nca_ranking(X, y)
        xgb_top_n = self._xgboost_ranking(X, y)
        l1_based_top_n = self._logReg_ranking(X, y)
        recursive_top_n = self._recursive_ranking(X, y)

        feature_sets = [nca_top_n, xgb_top_n, l1_based_top_n, recursive_top_n]
    
        def find_common_features(sets, count):
            """Finds features common to at least 'count' feature sets."""
            common_features = []
            
            for combo in itertools.combinations(sets, count):
                intersection = combo[0]
                for arr in combo[1:]:  # Iteratively find intersection across multiple sets
                    intersection = np.intersect1d(intersection, arr)
                common_features.append(intersection)

            return np.unique(np.concatenate(common_features)) if common_features else recursive_top_n

        # Handle different fusion criteria
        selected_features = find_common_features(feature_sets, self._min_common)

        self.logger.debug(f"Selected top {len(selected_features)} features: {selected_features}\n")
        
        return selected_features

    def decorator(method):
        """Decorator that checks the `func` argument and calls the appropriate method."""
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            func = kwargs.pop('func', None)
            if func is None:
                raise TypeError("Missing required argument 'func'. Expected a callable.")

            if callable(func):
                result = func(*args, **kwargs)
                return result
            else:
                raise TypeError(f"'func' must be callable. Got {type(func).__name__} instead.")
        return wrapper
    
    @decorator
    def fit(self):
        # Placeholder method for decorator function
        pass





    