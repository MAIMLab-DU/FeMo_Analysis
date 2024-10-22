
import time
import copy
import numpy as np
from tqdm import tqdm
from .base import FeMoBaseClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              AdaBoostClassifier,
                              GradientBoostingClassifier,
                              ExtraTreesClassifier,
                              VotingClassifier)
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV


class FeMoEnsembleClassifier(FeMoBaseClassifier):
    
    def __init__(self,
                 config = {
                     'search_space': {
                         'voting': ['soft']
                     },
                     'hyperparams': {
                         'logReg_clf': {'max_iter': 1000},
                         'rf_clf': {'n_estimators': 100},
                         'adaboost_clf': {'algorithm': 'SAMME.R'},
                         'gradboost_clf': {'subsample': 0.75, 'n_estimators': 1000},
                         'extraTrees_clf': {'n_estimators': 100},
                         'mlp_clf': {'hidden_layer_sizes': (110, ), 'max_iter': 1000}
                     }
                 }):
        super().__init__(config)
    
    def get_classifiers(self, params: dict, y: np.ndarray):
        class_weight = self._update_class_weight(y)
        return [
                ('logistic_regression', LogisticRegression(class_weight=class_weight,
                                                           random_state=0,
                                                           **params.get('logReg_clf'))),
                ('random_forest', RandomForestClassifier(class_weight=class_weight,
                                                         random_state=0,
                                                         n_jobs=-1,
                                                         criterion='entropy',
                                                         **params.get('rf_clf'))),
                ('adaboost', AdaBoostClassifier(random_state=0,
                                                **params.get('adaboost_clf'))),
                ('gradient_boosting', GradientBoostingClassifier(random_state=0,
                                                                 loss='exponential',
                                                                 **params.get('gradboost_clf'))),
                ('svc', SVC(probability=True, class_weight=class_weight, random_state=0)),
                ('knn', KNeighborsClassifier()),
                ('extra_trees', ExtraTreesClassifier(class_weight=class_weight, random_state=0,
                                                     n_jobs=-1, **params.get('extraTrees_clf'))),
                ('mlp', MLPClassifier(random_state=0, **params.get('mlp_clf')))
            ]

    # TODO: proper hyperparameter tuning - should tune individual classifiers
    # https://stackoverflow.com/questions/46580199/hyperparameter-in-voting-classifier
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

            classifiers = self.get_classifiers(hyperparams, y_train)
            voting_clf = VotingClassifier(estimators=classifiers, voting='soft', n_jobs=-1)
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)

            grid_search = GridSearchCV(
                estimator=voting_clf,
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

        self.hyperparams.update({'estimators': best_model.get_params().get('estimators')})
        self.hyperparams.update({'voting': best_model.get_params().get('voting')})
    
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

            if 'estimators' in hyperparams:
                classifiers = hyperparams['estimators']
            else:
                classifiers = self.get_classifiers(hyperparams, y_train)
            estimator = VotingClassifier(estimators=classifiers, voting='soft', n_jobs=-1)
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