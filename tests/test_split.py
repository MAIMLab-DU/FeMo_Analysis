import os
import joblib
import pytest
import numpy as np
from utils import (
    list_folders,
    compare_dictionaries
)
from data.dataset import DataProcessor

data_folder = "tests/datafiles"
folders = list_folders(data_folder)


@pytest.mark.parametrize("folder", folders)
def test_holdout_split(folder):
    """
    Test loading data from raw .dat files and 
    comparing it with pre-stored expected results.
    """
    
    data_processor = DataProcessor()
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
        strategy=True
    )

    desired_split_dict = joblib.load(
        os.path.join(data_folder, folder, "holdout.pkl")
    )

    compare_dictionaries(
        actual_dict=actual_split_dict,
        desired_dict=desired_split_dict
    )


# @pytest.mark.parametrize("folder", folders)
# def test_kfold_split(folder):
#     """
#     Test loading data from raw .dat files and 
#     comparing it with pre-stored expected results.
#     """
    
#     data_processor = DataProcessor()
#     extracted_features = joblib.load(
#         os.path.join(data_folder, folder, "extracted_features_train.pkl")
#     )
#     X_norm = joblib.load(
#         os.path.join(data_folder, folder, 'normalized_features.pkl')
#     )
#     top_feat_indices = joblib.load(
#         os.path.join(data_folder, folder, 'top_feat_indices.pkl')
#     )
#     X_norm = X_norm[:, top_feat_indices]

#     y_pre = extracted_features['labels']
#     actual_split_dict = data_processor.divide_data(
#         np.concatenate([X_norm, y_pre[:, np.newaxis]], axis=1),
#         num_folds=5
#     )

#     desired_split_dict = joblib.load(
#         os.path.join(data_folder, folder, "holdout.pkl")
#     )

#     compare_dictionaries(
#         actual_dict=actual_split_dict,
#         desired_dict=desired_split_dict
#     )


if __name__ == "__main__":
    pytest.main()

    