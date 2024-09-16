import numpy as np
from logger import LOGGER
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Result:
    best_model: object
    best_model_hyperparams: dict
    accuracy_scores: dict


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
        self.search_params: dict = config.get('search_params', {})
        self.fit_params: dict = config.get('fit_params', {})
        self.classifier = None

        self.result = Result(None, None, None)

    @staticmethod
    def _update_class_weight(y: np.ndarray, params: dict):
        
        num_tpd = np.sum(y == 1)
        num_fpd = np.sum(y == 0)

        class_weight: dict = {
            0: 1,
            1: num_fpd / num_tpd
        }
        params.update(class_weight=class_weight)

        return params

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
