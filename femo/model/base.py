import os
import joblib
import numpy as np
import pandas as pd
from ..logger import LOGGER
from collections import defaultdict
from typing import Literal
from keras.models import load_model
from dataclasses import dataclass, fields
from abc import ABC, abstractmethod
from .utils import stratified_kfold


@dataclass
class Result:
    accuracy_scores: dict = None
    preds: list[np.ndarray]|np.ndarray = None
    pred_scores: list[np.ndarray]|np.ndarray = None
    det_indices: list[np.ndarray]|np.ndarray = None
    filename_hash: list[np.ndarray]|np.ndarray = None

    @staticmethod
    def process_attributes(attribute,
                           metadata: dict,
                           preds: bool = False):
        tpd_attribute_rand = np.zeros((1, 1))
        fpd_attribute_rand = np.zeros((1, 1))

        num_folds = metadata['num_folds']
        for i in range(num_folds):
            # All except the last fold
            if i != num_folds - 1:
                tpd_attribute_rand = np.concatenate([tpd_attribute_rand, attribute[i][0:metadata['num_tpd_each_fold'], np.newaxis]])
                fpd_attribute_rand = np.concatenate([fpd_attribute_rand, attribute[i][metadata['num_tpd_each_fold']:, np.newaxis]])
            else:
                tpd_attribute_rand = np.concatenate([tpd_attribute_rand, attribute[i][0:metadata['num_tpd_last_fold'], np.newaxis]])
                fpd_attribute_rand = np.concatenate([fpd_attribute_rand, attribute[i][metadata['num_tpd_last_fold']:, np.newaxis]])

        # Remove the initial zeros added
        tpd_attribute_rand = tpd_attribute_rand[1:]
        fpd_attribute_rand = fpd_attribute_rand[1:]

        tpd_attribute = np.zeros((metadata['num_tpd'], 1))
        fpd_attribute = np.zeros((metadata['num_fpd'], 1))

        if num_folds > 1:  # Stratified K-fold division
            # Non-randomized the predictions to match with the original data set
            for i in range(metadata['num_tpd']):
                index = metadata['rand_num_tpd'][i]
                tpd_attribute[index] = tpd_attribute_rand[i]

            for i in range(metadata['num_fpd']):
                index = metadata['rand_num_fpd'][i]
                fpd_attribute[index] = fpd_attribute_rand[i]
        else:
            tpd_attribute = tpd_attribute_rand
            fpd_attribute = fpd_attribute_rand

        if preds:
            tpd_attribute = tpd_attribute >= 0.5
            fpd_attribute = fpd_attribute >= 0.5

        return np.concatenate([tpd_attribute, fpd_attribute])

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
                        metadata: dict,
                        results_path: str|None = None,
                        metadata_path: str|None = None):
        self._assert_no_none_fields()

        preds = self.process_attributes(self.preds, metadata, preds=True)
        pred_scores = self.process_attributes(self.pred_scores, metadata)
        det_indices = self.process_attributes(self.det_indices, metadata)
        filename_hash = self.process_attributes(self.filename_hash, metadata)

        self.preds = preds.astype(float)
        self.pred_scores = pred_scores.astype(float)
        self.det_indices = det_indices.astype(int)
        self.filename_hash = filename_hash.astype(int)

        if results_path is not None:
            results_path = os.path.join(results_path, "results.csv")
            self.save(results_path)
        
        if metadata_path is not None:
            metadata_path = os.path.join(metadata_path, "metadata.joblib")
            joblib.dump(metadata, metadata_path, compress=True)


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
        return f"{type(self)} hyperparameters: {self.hyperparams}\nsearch_space: {self.search_space}\nmodel: {self.classifier}"

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
    
    @staticmethod
    def split_data(data: np.ndarray,
                   strategy: Literal['holdout', 'kfold'] = 'holdout',
                   custom_ratio: int|None = None,
                   num_folds: int = 5):

        X, y = data[:, :-1], data[:, -1]
        train, test = [], []
        metadata = defaultdict()

        if strategy == 'holdout':
            raise NotImplementedError("TODO")

        # TODO: no mod from original implementation
        elif strategy == 'kfold':
            X_TPD_norm = X[y == 1]
            X_FPD_norm = X[y == 0]

            folds_dict = stratified_kfold(X_TPD_norm, X_FPD_norm, custom_ratio, num_folds)
            for k in range(num_folds):
                # Combine the k-th fold's X and Y to form the test set
                test.append(np.concatenate([folds_dict['X_K_fold'][k], folds_dict['Y_K_fold'][k][:, np.newaxis]], axis=1))
                
                # Stack all folds except the k-th fold to form the training set
                X_train_current = np.vstack([folds_dict['X_K_fold'][i] for i in range(num_folds) if i != k])
                Y_train_current = np.concatenate([folds_dict['Y_K_fold'][i] for i in range(num_folds) if i != k])

                train.append(np.concatenate([X_train_current, Y_train_current[:, np.newaxis]], axis=1))

        for key, val in folds_dict.items():
            if key not in ('X_K_fold', 'Y_K_fold'):
                metadata[key] = val
        
        return train, test, metadata
        
    def save_model(self,
                   model_filename: str,
                   model_framework: Literal['sklearn', 'keras']):
        if self.classifier is not None:
            try:
                if model_framework == 'sklearn':
                    assert model_filename.endswith('.pkl'), "Must be a pickle filename"
                    joblib.dump(self.classifier, model_filename)
                if model_framework == 'keras':
                    assert model_filename.endswith('.h5'), "Must be a h5 filename"
                    self.classifier.save(model_filename)
            except Exception as e:
                self.logger.error(f"Error saving model: {e}")
                raise Exception
        else:
            self.logger.error("No model trained yet. Cannot save.")

    def load_model(self,
                   model_filename: str,
                   model_framework: Literal['sklearn', 'keras']):
        try:
            if model_framework == 'sklearn':
                assert model_filename.endswith('.pkl'), "Must be a pickle filename"
                self.classifier = joblib.load(model_filename)
            if model_framework == 'keras':
                assert model_filename.endswith('.h5'), "Must be a h5 filename"
                self.classifier = load_model(model_filename)
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise Exception

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
    
    @abstractmethod
    def predict(self, *args, **kwargs):
        """Class method used to predict the target

        Raises:
            NotImplementedError
        """
        raise NotImplementedError
