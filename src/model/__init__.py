from .log_regression import FeMoLogRegClassifier
from .svm import FeMoSVClassifier
from .random_forest import FeMoRFClassifier
from .adaboost import FeMoAdaBoostClassifier
from .femonet import FeMoNNClassifier
from .ensemble import FeMoEnsembleClassifier


__all__ = [
    'FeMoLogRegClassifier',
    'FeMoSVClassifier',
    'FeMoRFClassifier',
    'FeMoAdaBoostClassifier',
    'FeMoNNClassifier',
    'FeMoEnsembleClassifier'
]

CLASSIFIER_MAP = {
    'logReg': FeMoLogRegClassifier,
    'svc': FeMoSVClassifier,
    'random-forest': FeMoRFClassifier,
    'adaboost': FeMoAdaBoostClassifier,
    'neural-net': FeMoNNClassifier,
    'ensemble': FeMoEnsembleClassifier
}