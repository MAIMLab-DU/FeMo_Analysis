import time
import copy
from tqdm import tqdm
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping
from scipy.special import expit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from .base import FeMoBaseClassifier
import tensorflow as tf
import warnings
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=DeprecationWarning)


# TODO: Same changes as FeMoEnsembleClassifier
class FeMoNeuralNet:

    def __init__(self,
                 num_hidden_layers: int,
                 units_per_layer: int,
                 dropout_rate: float = 0.25,
                 learning_rate: float = 1e-3) -> None:
        
        self.num_hidden_layers = num_hidden_layers
        self.units_per_layer = units_per_layer
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

    def compile_model(self, input_shape: tuple[int]):

        model = Sequential()
        
        # Add first layer with fixed units
        model.add(Dense(
            units=self.units_per_layer,
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

    model_framework = 'keras'

    def __init__(self,
                 config = {
                     'search_space': {
                         'num_hidden_layers': {
                             'low': 1,
                             'high': 5,
                             'step': 1
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
                         'units_per_layer': 10
                     }
                 }):
        super().__init__(config)

    # TODO: tune with optuna
    def tune(self,
               train_data: list[np.ndarray],
               test_data: list[np.ndarray],
               epochs: int = 10):
        
        start = time.time()

        assert len(train_data) == len(test_data), "Train, test data must have same folds"
        num_folds = len(train_data)
        units_per_layer = self.search_space.get('units_per_layer', {'low': 10, 'high': 200, 'step': 10})
        num_val_iterations = (units_per_layer['high']-units_per_layer['low']) // units_per_layer['step'] + 1
        hyperparams = copy.deepcopy(self.hyperparams)

        best_params = None
        best_accuracy = -1

        for j in tqdm(range(num_val_iterations), desc="Hyperparameter tuning..."):
            units_per_layer_value = units_per_layer['low'] + j * units_per_layer['step']
            hyperparams.update({'units_per_layer': units_per_layer_value})

            val_accuracies = []
            train_accuracies = []
            for k in range(num_folds):
                X_train, y_train = train_data[k][:, :-4], train_data[k][:, -1]
                X_val, y_val = test_data[k][:, :-4], test_data[k][:, -1]

                model = FeMoNeuralNet(**self.hyperparams).compile_model(
                    input_shape=(X_train.shape[1], )
                )
                model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    verbose=0,
                    class_weight={0:1, 1:2},
                    shuffle=False
                )

                # ---- Train accuracy -----
                y_hat_train = expit(model.predict(X_train))
                y_train_pred = (y_hat_train >= 0.5).astype(int)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                train_accuracies.append(train_accuracy)

                # ----- Validation accuracy -----
                y_hat_val = expit(model.predict(X_val))
                y_val_pred = (y_hat_val >= 0.5).astype(int)
                val_accuracy = accuracy_score(y_val, y_val_pred)
                val_accuracies.append(val_accuracy)
            
            if np.mean(val_accuracies) > best_accuracy:
                best_accuracy = np.mean(val_accuracies)
                best_params = hyperparams

            self.logger.info(f"Params: {hyperparams} - train: {np.mean(train_accuracies)} - validation: {np.mean(val_accuracies)}")
        
        self.logger.info(f"Hyperparam search took {time.time() - start: 0.2f} seconds")
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
            'test_f1_score': [],
            'roc_auc': []
        }

        for i in range(num_iterations):
            X_train, y_train = train_data[i][:, :-4], train_data[i][:, -1].astype(int)
            X_test, y_test = test_data[i][:, :-4], test_data[i][:, -1].astype(int)

            estimator = FeMoNeuralNet(**hyperparams).compile_model(
                input_shape=(X_train.shape[1], )
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

            y_hat_train = expit(estimator.predict(X_train))
            y_train_pred = (y_hat_train >= 0.5).astype(int)
            y_test = y_test[:, np.newaxis]
            y_hat_test = expit(estimator.predict(X_test))
            y_test_pred = (y_hat_test >= 0.5).astype(int)
            predictions.append(np.squeeze(y_test_pred))
            prediction_scores.append(np.squeeze(y_hat_test))            

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
            roc_auc = roc_auc_score(
                y_score=y_hat_test,
                y_true=y_test
            )
            eval_metrics['train_accuracy'].append(current_train_accuracy)
            eval_metrics['test_accuracy'].append(current_test_accuracy)
            eval_metrics['train_f1_score'].append(current_train_f1)
            eval_metrics['test_f1_score'].append(current_test_f1)
            eval_metrics['roc_auc'].append(roc_auc)

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
            self.logger.info(f"ROC AUC: {roc_auc:.3f}")
        
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

        pred_probabilities = expit(self.model.predict(X))
        pred_labels = (pred_probabilities >= 0.5).astype(int)

        return pred_labels
