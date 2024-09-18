import joblib
import numpy as np
from logger import LOGGER
from typing import Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Result:
    best_model_hyperparams: dict
    accuracy_scores: dict
    predictions: list[np.ndarray]


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

        self.result = Result(None, None, None)

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
    def search(self, *args, **kwargs):
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
