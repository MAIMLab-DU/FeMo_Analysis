import joblib
import numpy as np
import pandas as pd
from logger import LOGGER
from typing import Literal
from dataclasses import dataclass, fields
from abc import ABC, abstractmethod


@dataclass
class Result:
    accuracy_scores: dict = None
    preds: list[np.ndarray] = None
    pred_scores: list[np.ndarray] = None
    det_indices: list[np.ndarray] = None
    filename_hash: list[np.ndarray] = None

    def _assert_no_none_fields(self):
        # Check each field to ensure it is not None
        for field in fields(self):
            value = getattr(self, field.name)
            assert value is not None, f"Field '{field.name}' is None"
    
    def save(self, filename: str):
        results_df = {
            'filename_hash': [item for fold in self.filename_hash for item in fold],
            'det_indices': [item for fold in self.det_indices for item in fold],
            'predictions': [item for fold in self.preds for item in fold],
            'prediction_scores': [item for fold in self.pred_scores for item in fold]
        }
        results_df = pd.DataFrame(results_df)
        results_df.to_csv(filename, index=False)
    
    def compile_results(self,
                        features_df: pd.DataFrame|None = None,
                        filename: str|None = None):
        self._assert_no_none_fields()

        results_df = {
            'filename_hash': [item for sublist in self.filename_hash for item in sublist],
            'det_indices': [item for sublist in self.det_indices for item in sublist],
            'predictions': [item for sublist in self.preds for item in sublist],
            'prediction_scores': [item for sublist in self.pred_scores for item in sublist]
        }
        results_df = pd.DataFrame(results_df)

        if features_df is not None:
            results_df = features_df.merge(results_df, on=['filename_hash', 'det_indices'])        
        if filename is not None:
            results_df.to_csv(filename, index=False)
        
        return results_df


class FeMoBaseClassifier(ABC):

    @property
    def logger(self):
        return LOGGER
    
    @property
    def config(self) -> dict:
        return self._config

    def __init__(self,
                 config: dict):

        self._config = config
        self.search_space: dict = config.get('search_space', {})
        self.hyperparams: dict = config.get('hyperparams', {})
        self.classifier = None

        self.result = Result()
    
    def __repr__(self) -> str:
        return f"{type(self)} hyperparameters: {self.hyperparams}\nsearch_space: {self.search_space}"

    @staticmethod
    def _update_class_weight(y: np.ndarray, params: dict|None = None):
        
        num_tpd = np.sum(y == 1)
        num_fpd = np.sum(y == 0)

        class_weight: dict = {
            0: 1,
            1: num_fpd / num_tpd
        }
        if params is not None:
            params.update(class_weight=class_weight)
            return params
        else:
            return class_weight
        
    def save_model(self,
                   model_name: str,
                   model_framework: Literal['sklearn', 'keras']):
        if self.classifier is not None:
            try:
                if model_framework == 'sklearn':
                    joblib.dump(self.classifier, f"{model_name}.pkl")
                elif model_framework == 'keras':
                    self.classifier.save(f"{model_name}.h5")
            except Exception as e:
                self.logger.error(f"Error saving model: {e}")
        else:
            self.logger.error("No model trained yet. Cannot save.")

    @abstractmethod
    def tune(self, *args, **kwargs):
        """Class method using Cross Validation to search for best estimator 
        and hyperparameters

        Raises:
            NotImplementedError

        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, *args, **kwargs):
        """Class method used to fit the model

        Raises:
            NotImplementedError
        """
        raise NotImplementedError
