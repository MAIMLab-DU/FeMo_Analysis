import os
import joblib
import pytest
from utils import (
    list_folders,
    compare_elements,
    compare_dictionaries
)
from data.transforms import (
    DataLoader,
    DataPreprocessor,
    SensorFusion,
    DataSegmentor,
    DetectionExtractor,
    FeatureExtractor
)

data_folder = "tests/datafiles"
folders = list_folders(data_folder)


@pytest.mark.parametrize("folder", folders)
def test_load_data(folder):
    """
    Test loading data from raw .dat files and 
    comparing it with pre-stored expected results.
    """
    
    data_loader = DataLoader()
    actual_loaded_data = data_loader(
        os.path.join(data_folder, folder, "raw_data.dat")
    )
    desired_loaded_data = joblib.load(
        os.path.join(data_folder, folder, "loaded_data.pkl")
    )

    compare_dictionaries(
        actual_dict=actual_loaded_data, desired_dict=desired_loaded_data
    )


@pytest.mark.parametrize("folder", folders)
def test_preprocessed_data(folder):
    """
    Test preprocessing of data and 
    comparing the actual result with pre-stored expected results.
    """
    
    data_preprocessor = DataPreprocessor()
    loaded_data = joblib.load(os.path.join(data_folder, folder, "loaded_data.pkl"))
    actual_preprocessed_data = data_preprocessor(loaded_data=loaded_data)
    desired_preprocessed_data = joblib.load(
        os.path.join(data_folder, folder, "preprocessed_data.pkl")
    )

    compare_dictionaries(
        actual_dict=actual_preprocessed_data, desired_dict=desired_preprocessed_data
    )


@pytest.mark.parametrize("folder", folders)
def test_imu_map(folder):
    """
    Test imu_map from preprocessed data and 
    comparing it with pre-stored expected results.
    """

    data_segmentor = DataSegmentor()
    preprocessed_data = joblib.load(
        os.path.join(data_folder, folder, "preprocessed_data.pkl")
    )
    actual_imu_map = data_segmentor.transform(map_name='imu',
                                              preprocessed_data=preprocessed_data)
    desired_imu_map = joblib.load(os.path.join(data_folder, folder, "imu_map.pkl"))

    compare_elements(
        key='imu_map',
        actual=actual_imu_map,
        desired=desired_imu_map
    )


# TODO: add sensation_dict.pkl
@pytest.mark.parametrize("folder", folders)
def test_sensation_map(folder):
    """
    Test maternal sensation map from preprocessed data and 
    comparing it with pre-stored expected results.
    """

    data_segmentor = DataSegmentor()
    preprocessed_data = joblib.load(
        os.path.join(data_folder, folder, "preprocessed_data.pkl")
    )
    actual_sensation_map = data_segmentor.transform(map_name='sensation',
                                                preprocessed_data=preprocessed_data)
    desired_sensation_map = joblib.load(os.path.join(data_folder, folder, "sensation_map.pkl"))
    
    compare_elements(
        key='sensation_map',
        actual=actual_sensation_map,
        desired=desired_sensation_map
    )


@pytest.mark.parametrize("folder", folders)
def test_fm_map(folder):
    """
    Test fm_sensor map from preprocessed data and
    comparing it with pre-stored expected results.
    """

    data_segmentor = DataSegmentor()
    preprocessed_data = joblib.load(
        os.path.join(data_folder, folder, "preprocessed_data.pkl")
    )
    actual_fm_dict = data_segmentor.transform(map_name='fm_sensor',
                                              preprocessed_data=preprocessed_data)
    desired_fm_dict = joblib.load(os.path.join(data_folder, folder, "fm_dict.pkl"))

    compare_dictionaries(actual_dict=actual_fm_dict, desired_dict=desired_fm_dict)


@pytest.mark.parametrize("folder", folders)
def test_user_scheme(folder):
    """
    Test labeled fm map and 
    comparing it with pre-stored expected results.
    """

    data_fusion = SensorFusion()
    fm_dict = joblib.load(os.path.join(data_folder, folder, "fm_dict.pkl"))
    actual_user_scheme = data_fusion(fm_dict=fm_dict)
    desired_user_scheme = joblib.load(
        os.path.join(data_folder, folder, "scheme_dict.pkl")
    )

    compare_dictionaries(
        actual_dict=actual_user_scheme, desired_dict=desired_user_scheme
    )


@pytest.mark.parametrize("folder", folders)
def test_detections_for_train(folder):
    """
    Test detections extracted for train from preprocessed data and
    comparing it with pre-stored expected results.
    """
    
    detection_extractor = DetectionExtractor()

    preprocessed_data = joblib.load(
        os.path.join(data_folder, folder, "preprocessed_data.pkl")
    )
    scheme_dict = joblib.load(
        os.path.join(data_folder, folder, "scheme_dict.pkl")
    )
    sensation_map = joblib.load(
        os.path.join(data_folder, folder, "sensation_map.pkl")
    )
    actual_extracted_detections = detection_extractor.transform(
        inference=False,
        preprocessed_data=preprocessed_data, scheme_dict=scheme_dict,
        sensation_map=sensation_map
    )
    desired_extracted_detections = joblib.load(
        os.path.join(data_folder, folder, "extracted_detections_train.pkl")
    )

    compare_dictionaries(
        actual_dict=actual_extracted_detections,
        desired_dict=desired_extracted_detections,
    )


# TODO: test_detections_for_inference


@pytest.mark.parametrize("folder", folders)
def test_features_for_train(folder):
    """
    Test features extracted for inference from preprocessed data and
    comparing it with pre-stored expected results.
    """
    
    feature_extractor = FeatureExtractor()

    extracted_detections = joblib.load(
        os.path.join(data_folder, folder, "extracted_detections_train.pkl")
    )
    fm_dict = joblib.load(
        os.path.join(data_folder, folder, "fm_dict.pkl")
    )
    actual_extracted_features = feature_extractor.transform(
        inference=False,
        extracted_detections=extracted_detections, fm_dict=fm_dict
    )
    desired_extracted_features = joblib.load(
        os.path.join(data_folder, folder, "extracted_features_train.pkl")
    )

    compare_elements(
        key='features',
        actual=actual_extracted_features['features'],
        desired=desired_extracted_features['features']
    )
    compare_elements(
        key='labels',
        actual=actual_extracted_features['labels'],
        desired=desired_extracted_features['labels']
    )


# TODO: test_features_for_inference


if __name__ == "__main__":
    pytest.main()
