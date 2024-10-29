import numpy as np
import pandas as pd
from ..logger import LOGGER
from .utils import normalize_features
from .ranking import FeatureRanker


class Preprocessor(object):

    @property
    def logger(self):
        return LOGGER
    
    @property
    def feat_rank_cfg(self) -> dict:
        return self._feat_rank_cfg
    
    def __init__(self,
                 preprocess_config: dict = None
                ) -> None:
        
        self._feat_rank_cfg = preprocess_config.get('feature_ranking')
        self._feature_ranker = FeatureRanker(**self.feat_rank_cfg) if self.feat_rank_cfg is not None else FeatureRanker()

    def preprocess(self,
                   dataset: pd.DataFrame,
                   params_dict: dict = None) -> tuple[pd.DataFrame, dict]:
        self.logger.debug("Processing features...")
        if params_dict is None:
            params_dict = {
                'mu': None,
                'dev': None,
                'top_feat_indices': None
            }

        # Extract necessary columns from the input data
        X = dataset.drop(['labels', 'det_indices', 'filename_hash'], axis=1, errors='ignore').to_numpy()
        y_pre = dataset['labels'].to_numpy(dtype=int)
        det_indices = dataset['det_indices'].to_numpy(dtype=int)
        filename_hash = dataset['filename_hash'].to_numpy(dtype=int)

        X_norm, mu, dev = normalize_features(X, params_dict['mu'], params_dict['dev'])
        top_feat_indices = params_dict['top_feat_indices']
        if top_feat_indices is None:
            top_feat_indices = self._feature_ranker.fit(X_norm, y_pre, func=self._feature_ranker.ensemble_ranking)        

        X_norm = X_norm[:, top_feat_indices]
        columns = dataset.columns[top_feat_indices].tolist() + ['filename_hash', 'det_indices', 'labels']
        
        params_dict['mu'] = mu
        params_dict['dev'] = dev
        params_dict['top_feat_indices'] = top_feat_indices

        data = np.concatenate([X_norm, filename_hash[:, np.newaxis],
                               det_indices[:, np.newaxis], y_pre[:, np.newaxis]], axis=1)
        
        return pd.DataFrame(data, columns=columns), params_dict


