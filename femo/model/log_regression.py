import time
import copy
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from .base import FeMoBaseClassifier


# TODO: Same changes as FeMoEnsembleClassifier
class FeMoLogRegClassifier(FeMoBaseClassifier):

    model_framework = 'sklearn'

    def __init__(self,
                 config = {
                     'search_space': {
                         'Cs': 20
                     },
                     'hyperparams': {
                         'C': 1e-3
                     }
                 }):
        super().__init__(config)

    def tune(self,
               train_data: list[np.ndarray],
               test_data: list[np.ndarray]):
        
        start = time.time()

        assert len(train_data) == len(test_data), "Train, test data must have same folds"
        num_folds = len(train_data)

        best_model = None
        best_accuracy = -1
        
        for k in tqdm(range(num_folds), desc="Hyperparameter tuning..."):
            X_train, y_train = train_data[k][:, :-4], train_data[k][:, -1]
            X_val, y_val = test_data[k][:, :-4], test_data[k][:, -1]

            model = LogisticRegressionCV(
                cv=5,
                class_weight=self._update_class_weight(y_train),
                solver='lbfgs',
                max_iter=1000,
                verbose=False,
                n_jobs=-1,
                **self.search_space
            )

            model.fit(
                X_train, y_train
            )

            # ---- Train accuracy -----
            y_train_pred = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)

            # ----- Validation accuracy -----
            y_val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
        
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model

            self.logger.info(f"Model hyperparams: C: {model.C_[0]} - train: {train_accuracy} - validation: {val_accuracy}")
        
        self.logger.info(f"Hyperparameter tuning took {time.time() - start: 0.2f} seconds")
        self.logger.info(f"Best model hyperparameters: C: {best_model.C_[0]}")
        self.logger.info(f"Best model accuracy: {best_accuracy}")

        self.hyperparams.update({'C': best_model.C_[0]})

    def fit(self,
            train_data: list[np.ndarray],
            test_data: list[np.ndarray]):
        
        start = time.time()

        num_iterations = len(train_data)
        hyperparams = copy.deepcopy(self.hyperparams)
        
        best_f1_score = -1
        best_model = None
        predictions = []
        prediction_scores = []
        start_indices = []
        end_indices = []
        dat_file_keys = []
        eval_metrics = {
            'train_accuracy': [],
            'test_accuracy': [],
            'train_f1_score': [],
            'test_f1_score': []
        }

        for i in range(num_iterations):
            X_train, y_train = train_data[i][:, :-4], train_data[i][:, -1].astype(int)
            X_test, y_test = test_data[i][:, :-4], test_data[i][:, -1].astype(int)

            hyperparams = self._update_class_weight(y_train, hyperparams)
            estimator = LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                verbose=False,
                n_jobs=-1,
                **hyperparams
            )
            estimator.fit(X_train, y_train)

            y_train_pred = estimator.predict(X_train)
            y_test_pred = estimator.predict(X_test)
            y_test_pred_score = estimator.predict_proba(X_test)[:, 1]  # Score of Class 1
            predictions.append(np.squeeze(y_test_pred))
            prediction_scores.append(np.squeeze(y_test_pred_score))

            current_train_accuracy = accuracy_score(
                y_pred=y_train_pred,
                y_true=y_train
            )
            current_train_f1 = f1_score(
                y_true=y_train,
                y_pred=y_train_pred,
            )
            current_test_accuracy = accuracy_score(
                y_pred=y_test_pred,
                y_true=y_test
            )
            current_test_f1 = f1_score(
                y_true=y_test,
                y_pred=y_test_pred,
            )
            eval_metrics['train_accuracy'].append(current_train_accuracy)
            eval_metrics['test_accuracy'].append(current_test_accuracy)
            eval_metrics['train_f1_score'].append(current_train_f1)
            eval_metrics['test_f1_score'].append(current_test_f1)

            if current_test_f1 > best_f1_score:
                best_f1_score = current_test_f1
                best_model = estimator

            start_indices.append(test_data[i][:, -3])
            end_indices.append(test_data[i][:, -2])
            dat_file_keys.append(test_data[i][:, -4])

            self.logger.info(f"Iteration {i+1}:")
            self.logger.info(f"Training Accuracy: {current_train_accuracy:.3f}, F1 Score: {current_train_f1:.3f}")
            self.logger.info(f"Test Accuracy: {current_test_accuracy:.3f}, F1 Score: {current_test_f1:.3f}")
            self.logger.info(f"Best Test Accuracy: {best_f1_score:.3f}, Best F1 Score: {best_f1_score:.3f}")
        
        self.logger.info(f"Fitting model with train data took: {time.time() - start: 0.2f} seconds")
        self.logger.info(f"Average training accuracy: {np.mean(eval_metrics['train_accuracy'])}, F1 Score: {np.mean(eval_metrics['train_f1_score'])}")
        self.logger.info(f"Average testing accuracy: {np.mean(eval_metrics['test_accuracy'])}, F1 Score: {np.mean(eval_metrics['test_f1_score'])}")
        
        self.model = best_model
        self.result.accuracy_scores = eval_metrics
        self.result.preds = predictions
        self.result.pred_scores = prediction_scores
        self.result.start_indices = start_indices
        self.result.end_indices = end_indices
        self.result.dat_file_key = dat_file_keys

        return eval_metrics

    def predict(self, X):
        
        assert self.model is not None, "Error loading model"

        pred_labels = self.model.predict(X)

        return pred_labels
