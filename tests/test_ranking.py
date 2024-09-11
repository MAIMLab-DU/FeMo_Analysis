import os
import joblib
import pytest
import numpy as np
import pandas as pd
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
    
    df = pd.read_csv(os.path.join(data_folder, folder, 'features_ref.csv'), index_col=False)
    data_processor = DataProcessor(input_data=df)
    X_norm = data_processor._normalize_features(df.drop('labels', axis=1, errors='ignore').to_numpy())
    X_norm = X_norm[:, ~np.any(np.isnan(X_norm), axis=0)]
    y_pre = df.get('labels').to_numpy(dtype=int)
    feature_ranker = FeatureRanker()
    actual_feature_ranks = feature_ranker.fit(X_norm, y_pre, func=feature_ranker.ensemble_ranking)
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

    