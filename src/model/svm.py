import copy
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from .base import FeMoBaseClassifier


class FeMoSVClassifier(FeMoBaseClassifier):

    def __init__(self,
                 config = {
                     'search_space': {
                         'C': 20,
                         'gamma': 20
                     },
                     'hyperparams': {
                         'C': 1e-3,
                         'gamma': 'auto'
                     }
                 }):
        super().__init__(config)

        self.search_space['C'] = np.logspace(-4, 4, int(self.search_space['C'])).tolist()
        self.search_space['gamma'] = np.logspace(-4, 4, int(self.search_space['gamma'])).tolist()

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

            estimator = SVC(
                class_weight=self._update_class_weight(y_train),
                random_state=0,
                verbose=False,
                **hyperparams
            )
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
        
        self.logger.info(f"Hyperparameter tuning took {time.time() - start: 0.2f} seconds")
        self.logger.info(f"Best model hyperparameters: {best_model.get_params()}")
        self.logger.info(f"Best model accuracy: {best_accuracy}")

        for key in hyperparams.keys():
            self.hyperparams.update({key: best_model.get_params().get(key)})

    def fit(self,
            train_data: list[np.ndarray],
            test_data: list[np.ndarray]):
        
        start = time.time()

        num_iterations = len(train_data)

        hyperparams = copy.deepcopy(self.hyperparams)


        best_accuracy = -np.inf
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
            estimator = SVC(random_state=0, probability=True, **hyperparams)
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
        
        self.classifier = best_model
        self.result.accuracy_scores = accuracy_scores
        self.result.preds = predictions
        self.result.pred_scores = prediction_scores
        self.result.det_indices = det_indices
        self.result.filename_hash = filename_hash

