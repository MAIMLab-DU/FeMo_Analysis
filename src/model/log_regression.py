import time
import copy
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from .base import FeMoBaseClassifier


class FeMoLogRegClassifier(FeMoBaseClassifier):

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
            X_train, y_train = train_data[k][:, :-3], train_data[k][:, -1]
            X_val, y_val = test_data[k][:, :-3], test_data[k][:, -1]

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
        
        best_accuracy = -1
        best_model = None
        predictions = []
        prediction_scores = []
        det_indices = []
        filename_hash = []
        accuracy_scores = {
            'train_accuracy': [],
            'test_accuracy': []
        }

        for i in range(num_iterations):
            X_train, y_train = train_data[i][:, :-3], train_data[i][:, -1]
            X_test, y_test = test_data[i][:, :-3], test_data[i][:, -1]

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
            current_test_accuracy = accuracy_score(
                y_pred=y_test_pred,
                y_true=y_test
            )
            accuracy_scores['train_accuracy'].append(current_train_accuracy)
            accuracy_scores['test_accuracy'].append(current_test_accuracy)

            if current_test_accuracy > best_accuracy:
                best_accuracy = current_test_accuracy
                best_model = estimator

            det_indices.append(test_data[i][:, -2])
            filename_hash.append(test_data[i][:, -3])

            self.logger.info(f"Iteration {i+1}:")
            self.logger.info(f"Training Accuracy: {current_train_accuracy:.3f}")
            self.logger.info(f"Test Accuracy: {current_test_accuracy:.3f}")
            self.logger.info(f"Best Test Accuracy: {best_accuracy:.3f}")
        
        self.logger.info(f"Fitting model with train data took: {time.time() - start: 0.2f} seconds")
        self.logger.info(f"Average training accuracy: {np.mean(accuracy_scores['train_accuracy'])}")
        self.logger.info(f"Average testing accuracy: {np.mean(accuracy_scores['test_accuracy'])}")
        
        self.classifier = best_model
        self.result.accuracy_scores = accuracy_scores
        self.result.preds = predictions
        self.result.pred_scores = prediction_scores
        self.result.det_indices = det_indices
        self.result.filename_hash = filename_hash

    def predict(self, X):
        
        assert self.classifier is not None, "Error loading classifier"

        pred_labels = self.classifier.predict(X)

        return pred_labels
