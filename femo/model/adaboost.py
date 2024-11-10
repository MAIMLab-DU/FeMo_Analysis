import time
import copy
import numpy as np
from tqdm import tqdm
from .base import FeMoBaseClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV


class FeMoAdaBoostClassifier(FeMoBaseClassifier):

    def __init__(self,
                 config = {
                     'search_space': {
                         'n_estimators': [50, 100, 150,
                                          200, 250, 300],
                         'learning_rate': [0.01, 0.1, 0.5,
                                           1.0, 1.5, 2.0]
                     },
                     'hyperparams': {
                         'n_estimators': 50,
                         'learning_rate': 0.1
                     }
                 }):
        super().__init__(config)

    def tune(self,
               train_data: list[np.ndarray],
               test_data: list[np.ndarray]):
                
        start = time.time()

        assert len(train_data) == len(test_data), "Train, test data must have same folds"
        num_folds = len(train_data)
        hyperparams = copy.deepcopy(self.hyperparams)

        best_model = None
        best_accuracy = -1
        
        for k in tqdm(range(num_folds), desc="Hyperparameter tuning..."):
            X_train, y_train = train_data[k][:, :-3], train_data[k][:, -1]
            X_val, y_val = test_data[k][:, :-3], test_data[k][:, -1]

            base_estimator = GradientBoostingClassifier(n_estimators = 50, subsample = 0.75,
                                                        max_depth = 1, random_state = 42,
                                                        warm_start = True)
            estimator = AdaBoostClassifier(estimator=base_estimator, algorithm="SAMME.R",
                                           random_state=0, **hyperparams)
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)

            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=self.search_space,
                scoring='accuracy',
                cv=cv,
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(
                X_train, y_train
            )
            model = grid_search.best_estimator_

            # ---- Train accuracy -----
            y_train_pred = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)

            # ----- Validation accuracy -----
            y_val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
        
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model

            self.logger.info(f"Model hyperparams: {model.get_params()} - train: {train_accuracy} - validation: {val_accuracy}")
        
        self.logger.info(f"Hyperparam search took {time.time() - start: 0.2f} seconds")
        self.logger.info(f"Best model hyperparameters: {best_model.get_params()}")
        self.logger.info(f"Best model accuracy: {best_accuracy}")
        for key in self.hyperparams.keys():
            self.hyperparams.update({key: best_model.get_params().get(key)})
        self.hyperparams.update({'estimator': best_model.get_params().get('estimator')})

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

            estimator = AdaBoostClassifier(algorithm='SAMME.R', random_state=0,
                                           **hyperparams)
            estimator.fit(X_train, y_train)

            y_train_pred = estimator.predict(X_train)
            y_test_pred = estimator.predict(X_test)
            y_test_pred_score = estimator.predict_proba(X_test)[:, 1]  # Index of class 1
            predictions.append(y_test_pred)
            prediction_scores.append(y_test_pred_score)

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
        
        self.model = best_model
        self.result.accuracy_scores = accuracy_scores
        self.result.preds = predictions
        self.result.pred_scores = prediction_scores
        self.result.det_indices = det_indices
        self.result.filename_hash = filename_hash

    def predict(self, X):
        
        assert self.model is not None, "Error loading model"

        pred_labels = self.model.predict(X)

        return pred_labels