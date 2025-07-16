import os
import joblib
import numpy as np
import pandas as pd
from ..logger import LOGGER
from collections import defaultdict
from typing import Literal, Dict, Any
from keras.models import load_model
from dataclasses import dataclass, fields
from abc import ABC, abstractmethod
from .utils import stratified_kfold


@dataclass
class Result:
    accuracy_scores: dict = None
    preds: list[np.ndarray]|np.ndarray = None
    pred_scores: list[np.ndarray]|np.ndarray = None
    start_indices: list[np.ndarray]|np.ndarray = None
    end_indices: list[np.ndarray]|np.ndarray = None
    dat_file_key: list[np.ndarray]|np.ndarray = None

    def _assert_no_none_fields(self):
        # Check each field to ensure it is not None
        for field in fields(self):
            value = getattr(self, field.name)
            assert value is not None, f"Field '{field.name}' is None"
    
    def save(self, filename: str):
        results_df = {
            'dat_file_key': [item for fold in self.dat_file_key for item in fold] 
                            if isinstance(self.dat_file_key, list) else np.squeeze(self.dat_file_key).tolist(),
            'start_indices': [item for fold in self.start_indices for item in fold] 
                            if isinstance(self.start_indices, list) else np.squeeze(self.start_indices).tolist(),
            'end_indices': [item for fold in self.end_indices for item in fold]
                            if isinstance(self.end_indices, list) else np.squeeze(self.end_indices).tolist(),
            'predictions': [item for fold in self.preds for item in fold] 
                            if isinstance(self.preds, list) else np.squeeze(self.preds).tolist(),
            'prediction_scores': [item for fold in self.pred_scores for item in fold] 
                            if isinstance(self.pred_scores, list) else np.squeeze(self.pred_scores).tolist()
        }
        results_df = pd.DataFrame(results_df)
        results_df.to_csv(filename, index=False)
    
    def compile_results(self,
                        metadata: dict,
                        threshold: float = 0.5,
                        results_path: str|None = None,
                        metadata_path: str|None = None):
        self._assert_no_none_fields()

        if results_path is not None:
            results_path = os.path.join(results_path, "results.csv")
            self.save(results_path)
        
        if metadata_path is not None:
            metadata_path = os.path.join(metadata_path, "metadata.joblib")
            joblib.dump(metadata, metadata_path, compress=True)


class FeMoBaseClassifier(ABC):

    model_framework: Literal['sklearn', 'keras'] = None

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
        self.model = None

        self.result = Result()
    
    def __repr__(self) -> str:
        return f"{type(self)} hyperparameters: {self.hyperparams}\nsearch_space: {self.search_space}\nmodel: {self.model}"

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

        X, y = data[:, :-1], data[:, -1].astype(int)
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
                   model_filename: str):
        if self.model is not None:
            try:
                if self.model_framework == 'sklearn':
                    assert model_filename.endswith('.pkl') or model_filename.endswith('.joblib'), "Must be a pickle or joblib file"
                    joblib.dump(self.model, model_filename)
                if self.model_framework == 'keras':
                    model_filename = model_filename.replace('.joblib', '.h5').replace('.pkl', '.h5')
                    assert model_filename.endswith('.h5'), "Must be a h5 file"
                    self.model.save(model_filename)
            except Exception as e:
                self.logger.error(f"Error saving model: {e}")
                raise Exception
        else:
            self.logger.error("No model trained yet. Cannot save.")

    def load_model(self,
                   model_filename: str):
        try:
            if self.model_framework == 'sklearn':
                assert model_filename.endswith('.pkl') or model_filename.endswith('.joblib'), "Must be a pickle or joblib file"
                self.model = joblib.load(model_filename)
            if self.model_framework == 'keras':
                model_filename = model_filename.replace('.joblib', '.h5').replace('.pkl', '.h5')
                assert model_filename.endswith('.h5'), "Must be a h5 file"
                self.model = load_model(model_filename)
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
    def fit(self, *args, **kwargs) -> Dict[str, Any]:
        """Class method used to fit the model
        
        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics with string keys
                        and numeric values (accuracy, f1-score, etc.)

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, *args, **kwargs):
        """Class method used to predict the target

        Raises:
            NotImplementedError
        """
        raise NotImplementedError
