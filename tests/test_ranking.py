import os
import joblib
import pytest
from utils import (
    list_folders,
    compare_elements
)
from data.ranking import FeatureRanker
from data.dataset import DataProcessor

data_folder = "tests/datafiles"
folders = list_folders(data_folder)


@pytest.mark.parametrize("folder", folders)
def test_feature_ranking(folder):
    """
    Test loading data from raw .dat files and 
    comparing it with pre-stored expected results.
    """
    
    extracted_features = joblib.load(
        os.path.join(data_folder, folder, "extracted_features_train.pkl")
    )
    feature_ranker = FeatureRanker()
    data_processor = DataProcessor()
    X_norm = extracted_features['features']
    X_norm = data_processor._normalize_features(X_norm)
    desired_norm_features = joblib.load(
        os.path.join(data_folder, folder, 'normalized_features.pkl')
    )
    compare_elements(
        key='norm_features',
        actual=X_norm,
        desired=desired_norm_features
    )

    y_pre = extracted_features['labels']

    actual_feature_ranks = feature_ranker.fit(X_norm, y_pre,
                                              func=feature_ranker.ensemble_feature_selection)
    desired_feature_ranks = joblib.load(
        os.path.join(data_folder, folder, "top_feat_indices.pkl")
    )

    compare_elements(
        key='feature_ranks',
        actual=actual_feature_ranks,
        desired=desired_feature_ranks
    )


if __name__ == "__main__":
    pytest.main()

    