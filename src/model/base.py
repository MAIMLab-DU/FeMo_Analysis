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
    preds: list[np.ndarray]|np.ndarray = None
    pred_scores: list[np.ndarray]|np.ndarray = None
    det_indices: list[np.ndarray]|np.ndarray = None
    filename_hash: list[np.ndarray]|np.ndarray = None

    @staticmethod
    def process_attributes(attribute,
                                   split_dict: dict,
                                   nn_pred: bool,
                                   preds: bool = False):
        tpd_attribute_rand = np.zeros((1, 1))
        fpd_attribute_rand = np.zeros((1, 1))

        num_folds = len(split_dict['test'])
        for i in range(num_folds):
            # All except the last fold
            if i != num_folds - 1:
                tpd_attribute_rand = np.concatenate([tpd_attribute_rand, attribute[i][0:split_dict['num_tpd_each_fold'], np.newaxis]])
                fpd_attribute_rand = np.concatenate([fpd_attribute_rand, attribute[i][split_dict['num_tpd_each_fold']:, np.newaxis]])
            else:
                tpd_attribute_rand = np.concatenate([tpd_attribute_rand, attribute[i][0:split_dict['num_tpd_last_fold'], np.newaxis]])
                fpd_attribute_rand = np.concatenate([fpd_attribute_rand, attribute[i][split_dict['num_tpd_last_fold']:, np.newaxis]])

        if nn_pred:
            # Remove the initial zeros added
            tpd_attribute_rand = tpd_attribute_rand[1:]
            fpd_attribute_rand = fpd_attribute_rand[1:]

            tpd_attribute = np.zeros((split_dict['num_tpd'], 1))
            fpd_attribute = np.zeros((split_dict['num_fpd'], 1))

            if num_folds > 1:  # Stratified K-fold division
                # Non-randomized the predictions to match with the original data set
                for i in range(split_dict['num_tpd']):
                    index = split_dict['rand_num_tpd'][i]
                    tpd_attribute[index] = tpd_attribute_rand[i]

                for i in range(split_dict['num_fpd']):
                    index = split_dict['rand_num_fpd'][i]
                    fpd_attribute[index] = fpd_attribute_rand[i]
            else:
                tpd_attribute = tpd_attribute_rand
                fpd_attribute = fpd_attribute_rand

            if preds:
                tpd_attribute = tpd_attribute >= 0.5
                fpd_attribute = fpd_attribute >= 0.5

            return np.concatenate([tpd_attribute, fpd_attribute])
        else:
            # TODO: for other models
            raise NotImplementedError("TODO")

    def _assert_no_none_fields(self):
        # Check each field to ensure it is not None
        for field in fields(self):
            value = getattr(self, field.name)
            assert value is not None, f"Field '{field.name}' is None"
    
    def save(self, filename: str):
        results_df = {
            'filename_hash': [item for fold in self.filename_hash for item in fold] 
                            if isinstance(self.filename_hash, list) else np.squeeze(self.filename_hash).tolist(),
            'det_indices': [item for fold in self.det_indices for item in fold] 
                            if isinstance(self.det_indices, list) else np.squeeze(self.det_indices).tolist(),
            'predictions': [item for fold in self.preds for item in fold] 
                            if isinstance(self.preds, list) else np.squeeze(self.preds).tolist(),
            'prediction_scores': [item for fold in self.pred_scores for item in fold] 
                            if isinstance(self.pred_scores, list) else np.squeeze(self.pred_scores).tolist()
        }
        results_df = pd.DataFrame(results_df)
        results_df.to_csv(filename, index=False)
    
    def compile_results(self,
                        split_dict: dict,
                        nn_pred: bool = False,
                        filename: str|None = None):
        self._assert_no_none_fields()

        preds = self.process_attributes(self.preds, split_dict, nn_pred, preds=True)
        pred_scores = self.process_attributes(self.pred_scores, split_dict, nn_pred)
        det_indices = self.process_attributes(self.det_indices, split_dict, nn_pred)
        filename_hash = self.process_attributes(self.filename_hash, split_dict, nn_pred)

        self.preds = preds.astype(float)
        self.pred_scores = pred_scores.astype(float)
        self.det_indices = det_indices.astype(int)
        self.filename_hash = filename_hash.astype(int)

        if filename is not None:
            self.save(filename)


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
