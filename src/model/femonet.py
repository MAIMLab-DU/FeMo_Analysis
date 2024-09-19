import time
import copy
import optuna
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping
from scipy.special import expit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from .base import FeMoBaseClassifier
import tensorflow as tf
import warnings
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=DeprecationWarning)


class FeMoNeuralNet:
    def __init__(self,
                 num_hidden_layers: int,
                 units_first_layer: int,
                 units_per_layer: int,
                 dropout_rate: float = 0.25,
                 learning_rate: float = 1e-3) -> None:
        
        self.num_hidden_layers = num_hidden_layers
        self.units_first_layer = units_first_layer
        self.units_per_layer = units_per_layer
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

    def compile_model(self, input_shape: tuple[int]):

        model = Sequential()
        
        # Add first layer with fixed units
        model.add(Dense(
            units=self.units_first_layer,
            input_shape=input_shape,
            activation='relu',
            kernel_initializer='he_uniform'
        ))
        model.add(Dropout(self.dropout_rate))
        
        # Add hidden layers with tunable units
        for _ in range(self.num_hidden_layers):
            model.add(Dense(
                units=self.units_per_layer,
                activation='relu'
            ))
        
        # Output layer
        model.add(Dense(1, activation='linear'))

        # Compile model
        model.compile(
            loss=BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate  # Fixed learning rate
            ),
            metrics=['accuracy']
        )
        
        return model


class FeMoNNClassifier(FeMoBaseClassifier):

    def __init__(self,
                 config = {
                     'search_space': {
                         'num_hidden_layers': {
                             'low': 1,
                             'high': 5,
                             'step': 1
                         },
                         'units_first_layer': {
                             'low': 10,
                             'high': 200,
                             'step': 10
                         },
                         'units_per_layer': {
                             'low': 10,
                             'high': 200,
                             'step': 10
                         },
                         'dropout_rate': {
                             'low': 0.1,
                             'high': 0.5,
                             'step': 0.05
                         },
                         'learning_rate': {
                             'low': 1e-4,
                             'high': 1e-2,
                             'log': True
                         }
                     },
                     'hyperparams': {
                         'num_hidden_layers': 1, 
                         'units_first_layer': 10,
                         'units_per_layer': 10
                     }
                 }):
        super().__init__(config)

    def search(self,
               train_data: list[np.ndarray],
               test_data: list[np.ndarray],
               epochs: int = 10,
               patience: int = 5,
               n_trials: int = 10):
        
        start = time.time()

        assert len(train_data) == len(test_data), "Train, test data must have same folds"
        num_folds = len(train_data)

        def objective(trial: optuna.Trial):

            params = {
                'num_hidden_layers': trial.suggest_int('num_hidden_layers', **self.search_space['num_hidden_layers']),
                'units_first_layer': trial.suggest_int('units_first_layer', **self.search_space['units_first_layer']),
                'units_per_layer': trial.suggest_int('units_per_layer', **self.search_space['units_per_layer']),
                'dropout_rate': trial.suggest_float('dropout_rate', **self.search_space['dropout_rate']),
                'learning_rate': trial.suggest_float('learning_rate', **self.search_space['learning_rate'])
            }
            cv_inner = KFold(n_splits=5, shuffle=True, random_state=42)

            accuracy_scores = []
            for i in range(num_folds):
                X_train, y_train = train_data[i][:, :-1], train_data[i][:, -1]

                cv_scores = []
                for train_idx, val_idx in cv_inner.split(X_train, y_train):
                    X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
                    X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

                    estimator = FeMoNeuralNet(**params).compile_model(
                        input_shape=(train_data[0].shape[1] - 1, )
                    )
                    estimator.fit(
                        X_train_fold,
                        y_train_fold,
                        epochs=epochs,
                        shuffle=False,
                        verbose=0,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=patience)]
                    )
                    y_hat_val = expit(estimator.predict(X_val_fold))
                    y_val_pred = (y_hat_val >= 0.5).astype(int)
                    cv_scores.append(
                        accuracy_score(y_val_pred, y_val_fold)
                    )
                accuracy_scores.append(np.mean(cv_scores))
            
            return np.mean(accuracy_scores)
        
        # By default, optuna uses TPE sampling
        study = optuna.create_study(direction='maximize')
        self.logger.info(f"Performing Grid Search with {num_folds}x5 Cross-validation")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_accuracy = study.best_value
        
        self.logger.info(f"Optuna search took {time.time() - start: 0.2f} seconds")
        self.logger.info(f"Best model hyperparameters: {best_params}")
        self.logger.info(f"Best model accuracy: {best_accuracy}")

        for key, value in best_params.items():
            self.hyperparams.update({key: value})

    def fit(self,
            train_data: list[np.ndarray],
            test_data: list[np.ndarray],
            epochs: int = 10,
            patience: int = 5):
        
        start = time.time()

        num_iterations = len(train_data)
        hyperparams = copy.deepcopy(self.hyperparams)
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True
        ) 
        
        best_accuracy = -1
        best_model = None
        predictions = []
        prediction_scores = []
        accuracy_scores = {
            'train_accuracy': [],
            'test_accuracy': []
        }

        for i in range(num_iterations):
            X_train, y_train = train_data[i][:, :-1], train_data[i][:, -1]
            X_test, y_test = test_data[i][:, :-1], test_data[i][:, -1]

            estimator = FeMoNeuralNet(**hyperparams).compile_model(
                input_shape=(train_data[0].shape[1] - 1, )
            )
            estimator.fit(
                x=X_train,
                y=y_train,
                validation_split=0.2,
                # class_weight=self._update_class_weight(y_train),
                class_weight={0: 1, 1: 2},
                callbacks=[early_stopping],
                epochs=epochs,
                shuffle=False,
                verbose=0
            )

            # Train metrics
            y_hat_train = expit(estimator.predict(X_train))
            y_train_pred = (y_hat_train >= 0.5).astype(int)
            current_train_accuracy = accuracy_score(
                y_pred=y_train_pred,
                y_true=y_train[:, np.newaxis]
            )

            # Test metrics
            y_hat_test = expit(estimator.predict(X_test))
            y_test_pred = (y_hat_test >= 0.5).astype(int)
            predictions.append(np.squeeze(y_test_pred))
            prediction_scores.append(np.squeeze(y_hat_test))
            
            current_test_accuracy = accuracy_score(
                y_pred=y_test_pred,
                y_true=y_test[:, np.newaxis]
            )
            roc_auc = roc_auc_score(
                y_score=y_hat_test,
                y_true=y_test[:, np.newaxis]
            )

            accuracy_scores['train_accuracy'].append(current_train_accuracy)
            accuracy_scores['test_accuracy'].append(current_test_accuracy)

            if current_test_accuracy > best_accuracy:
                best_accuracy = current_test_accuracy
                best_model = estimator

            self.logger.info(f"Iteration {i+1}:")
            self.logger.info(f"Training Accuracy: {current_train_accuracy:.3f}")
            self.logger.info(f"Test Accuracy: {current_test_accuracy:.3f}")
            self.logger.info(f"ROC-AUC Score: {roc_auc:.3f}")
            self.logger.info(f"Best Test Accuracy: {best_accuracy:.3f}")
        
        self.logger.info(f"Fitting model with train data took: {time.time() - start: 0.2f} seconds")
        self.logger.info(f"Average training accuracy: {np.mean(accuracy_scores['train_accuracy'])}")
        self.logger.info(f"Average testing accuracy: {np.mean(accuracy_scores['test_accuracy'])}")
        
        self.classifier = best_model
        self.result.accuracy_scores = accuracy_scores
        self.result.preds = predictions
        self.result.pred_scores = prediction_scores
