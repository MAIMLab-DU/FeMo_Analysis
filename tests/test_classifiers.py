# TODO: implement functionality
import os
import joblib
import pytest
import numpy as np
from utils import (
    list_folders
)
from data.dataset import DataProcessor
from model import (
    FeMoLogRegClassifier,
    FeMoSVClassifier,
    FeMoRFClassifier,
    FeMoAdaBoostClassifier,
    FeMoNNClassifier
)

data_folder = "tests/datafiles"
folders = list_folders(data_folder)
strategies = ['holdout', 'kfold']


@pytest.mark.parametrize("folder", folders)
@pytest.mark.parametrize("strategy", strategies)
def test_logReg_classifier(folder, strategy):
    """
    Test loading data from raw .dat files and 
    comparing it with pre-stored expected results.
    """
    
    data_processor = DataProcessor()
    classifier = FeMoLogRegClassifier()
    extracted_features = joblib.load(
        os.path.join(data_folder, folder, "extracted_features_train.pkl")
    )
    X_norm = joblib.load(
        os.path.join(data_folder, folder, 'normalized_features.pkl')
    )
    top_feat_indices = joblib.load(
        os.path.join(data_folder, folder, 'top_feat_indices.pkl')
    )
    X_norm = X_norm[:, top_feat_indices]

    y_pre = extracted_features['labels']
    actual_split_dict = data_processor.split_data(
        np.concatenate([X_norm, y_pre[:, np.newaxis]], axis=1),
        strategy=strategy
    )
    train_data, test_data = actual_split_dict['train'], actual_split_dict['test']

    classifier.tune(train_data, test_data)
    classifier.fit(train_data, test_data)
    result = classifier.result
    print(result)
        
        # TODO: assert with actual results
    assert isinstance(result.accuracy_scores, dict)


@pytest.mark.parametrize("folder", folders)
@pytest.mark.parametrize("strategy", strategies)
def test_svm_classifier(folder, strategy):
    """
    Test loading data from raw .dat files and 
    comparing it with pre-stored expected results.
    """
    
    data_processor = DataProcessor()
    classifier = FeMoSVClassifier()
    extracted_features = joblib.load(
        os.path.join(data_folder, folder, "extracted_features_train.pkl")
    )
    X_norm = joblib.load(
        os.path.join(data_folder, folder, 'normalized_features.pkl')
    )
    top_feat_indices = joblib.load(
        os.path.join(data_folder, folder, 'top_feat_indices.pkl')
    )
    X_norm = X_norm[:, top_feat_indices]

    y_pre = extracted_features['labels']
    actual_split_dict = data_processor.split_data(
        np.concatenate([X_norm, y_pre[:, np.newaxis]], axis=1),
        strategy=strategy
    )
    train_data, test_data = actual_split_dict['train'], actual_split_dict['test']

    classifier.tune(train_data, test_data)
    classifier.fit(train_data, test_data)
    result = classifier.result
    
    # TODO: assert with actual results
    assert isinstance(result.accuracy_scores, dict)


@pytest.mark.parametrize("folder", folders)
@pytest.mark.parametrize("strategy", strategies)
def test_rf_classifier(folder, strategy):
    """
    Test loading data from raw .dat files and 
    comparing it with pre-stored expected results.
    """
    
    data_processor = DataProcessor()
    classifier = FeMoRFClassifier()
    extracted_features = joblib.load(
        os.path.join(data_folder, folder, "extracted_features_train.pkl")
    )
    X_norm = joblib.load(
        os.path.join(data_folder, folder, 'normalized_features.pkl')
    )
    top_feat_indices = joblib.load(
        os.path.join(data_folder, folder, 'top_feat_indices.pkl')
    )
    X_norm = X_norm[:, top_feat_indices]

    y_pre = extracted_features['labels']
    actual_split_dict = data_processor.split_data(
        np.concatenate([X_norm, y_pre[:, np.newaxis]], axis=1),
        strategy=strategy
    )
    train_data, test_data = actual_split_dict['train'], actual_split_dict['test']

    classifier.tune(train_data, test_data)
    classifier.fit(train_data, test_data)
    result = classifier.result
    print(result)
    
    # TODO: assert with actual results
    assert isinstance(result.accuracy_scores, dict)


@pytest.mark.parametrize("folder", folders)
@pytest.mark.parametrize("strategy", strategies)
def test_adaboost_classifier(folder, strategy):
    """
    Test loading data from raw .dat files and 
    comparing it with pre-stored expected results.
    """
    
    data_processor = DataProcessor()
    classifier = FeMoAdaBoostClassifier()
    extracted_features = joblib.load(
        os.path.join(data_folder, folder, "extracted_features_train.pkl")
    )
    X_norm = joblib.load(
        os.path.join(data_folder, folder, 'normalized_features.pkl')
    )
    top_feat_indices = joblib.load(
        os.path.join(data_folder, folder, 'top_feat_indices.pkl')
    )
    X_norm = X_norm[:, top_feat_indices]

    y_pre = extracted_features['labels']
    actual_split_dict = data_processor.split_data(
        np.concatenate([X_norm, y_pre[:, np.newaxis]], axis=1),
        strategy=strategy
    )
    train_data, test_data = actual_split_dict['train'], actual_split_dict['test']

    classifier.tune(train_data, test_data)
    classifier.fit(train_data, test_data)
    result = classifier.result
    print(result)
    
    # TODO: assert with actual results
    assert isinstance(result.accuracy_scores, dict)


@pytest.mark.parametrize("folder", folders)
@pytest.mark.parametrize("strategy", strategies)
def test_femonet_classifier(folder, strategy):
    """
    Test loading data from raw .dat files and 
    comparing it with pre-stored expected results.
    """
    
    data_processor = DataProcessor()
    classifier = FeMoNNClassifier()
    extracted_features = joblib.load(
        os.path.join(data_folder, folder, "extracted_features_train.pkl")
    )
    X_norm = joblib.load(
        os.path.join(data_folder, folder, 'normalized_features.pkl')
    )
    top_feat_indices = joblib.load(
        os.path.join(data_folder, folder, 'top_feat_indices.pkl')
    )
    X_norm = X_norm[:, top_feat_indices]

    y_pre = extracted_features['labels']
    actual_split_dict = data_processor.split_data(
        np.concatenate([X_norm, y_pre[:, np.newaxis]], axis=1),
        strategy=strategy
    )
    train_data, test_data = actual_split_dict['train'], actual_split_dict['test']

    classifier.tune(train_data, test_data)
    classifier.fit(train_data, test_data)
    result = classifier.result
    print(result)
    
    # TODO: assert with actual results
    assert isinstance(result.accuracy_scores, dict)


if __name__ == "__main__":
    pytest.main()

    