import unittest
from unittest import TestCase
import os
import joblib
from utils import list_folders, compare_dictionaries
from data.extraction import DetectionExtractor


class TestInferenceDetections(TestCase):
    data_folder = 'tests/datafiles'

    detection_extractor = DetectionExtractor(
        base_dir=data_folder,
        inferece=True
    )

    def test_user_scheme(self):
        
        folders = list_folders(self.data_folder)

        for folder in folders:

            preprocessed_data = joblib.load(
                os.path.join(self.data_folder, folder, "preprocessed_data.pkl")
            )
            scheme_dict = joblib.load(
                os.path.join(self.data_folder, folder, "labeled_user_scheme.pkl")
            )
            actual_extracted_detections = self.detection_extractor.extract_detections(
                preprocessed_data=preprocessed_data,
                scheme_dict=scheme_dict
            )
            desired_extracted_detections = joblib.load(
                os.path.join(self.data_folder, folder, "extracted_detections.pkl")
            )

            compare_dictionaries(
                actual_dict=actual_extracted_detections,
                desired_dict=desired_extracted_detections
            )


if __name__ == '__main__':
    unittest.main()