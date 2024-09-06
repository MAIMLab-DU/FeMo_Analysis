import unittest
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from data.segmentation import DataSegmentor


class TestSegmenation(TestCase):
    data_folder = 'tests/datafiles'
    input_folder = os.path.join(data_folder, 'input')
    output_folder = os.path.join(data_folder, 'output')

    data_segmentor = DataSegmentor(
        base_dir=data_folder
    )

    def test_imu_map(self):

        for folder in os.listdir(self.input_folder):
            preprocessed_data = defaultdict()

            for file in os.listdir(os.path.join(self.input_folder, folder)):
                if file.endswith('.npz') and file.startswith('preprocessed-'):
                    key = file.split('.npz')[0].split('preprocessed-')[-1]
                    preprocessed_data[key] = np.load(os.path.join(self.input_folder, folder, file))['arr_0']
                if file.endswith('.pkl') and file.startswith('preprocessed-'):
                    key = file.split('.pkl')[0].split('preprocessed-')[-1]
                    preprocessed_data[key] = pd.read_pickle(os.path.join(self.input_folder, folder, file))
                    
            imu_map = self.data_segmentor.create_imu_map(preprocessed_data)
            out_file = os.path.join(self.output_folder, 
                                    folder, 
                                    f"imu_map.npz")
            output_data = np.load(out_file)['arr_0']

            np.testing.assert_array_equal(
                imu_map,  # function output
                output_data  # expected output
            )
    
    # TODO: add test implementation
    def test_fm_map(self):
        preprocessed_data = defaultdict()

        for folder in os.listdir(self.input_folder):
            preprocessed_data = defaultdict()

            for file in os.listdir(os.path.join(self.input_folder, folder)):
                if file.endswith('.npz') and file.startswith('preprocessed-'):
                    key = file.split('.npz')[0].split('preprocessed-')[-1]
                    preprocessed_data[key] = np.load(os.path.join(self.input_folder, folder, file))['arr_0']
                if file.endswith('.pkl') and file.startswith('preprocessed-'):
                    key = file.split('.pkl')[0].split('preprocessed-')[-1]
                    preprocessed_data[key] = pd.read_pickle(os.path.join(self.input_folder, folder, file))
            
            fm_dict = self.data_segmentor.create_fm_map(preprocessed_data)
            for k, v in fm_dict.items():
                out_file = os.path.join(self.output_folder, 
                                        folder, 
                                        f"{k}.npz")
                output_data = np.load(out_file)['arr_0']
                if output_data.dtype == bool:
                    np.testing.assert_array_equal(
                        v,  # function output
                        output_data,  # expected output
                        err_msg=f"{k} mismatch"
                    )
                if output_data.dtype == float:
                    np.testing.assert_allclose(
                                actual=v,  # your function output
                                desired=output_data,  # the expected output
                                rtol=1e-8,  # relative tolerance
                                atol=1e-12,  # absolute tolerance
                                err_msg=f"{k} mismatch"
                            )


if __name__ == '__main__':
    unittest.main()