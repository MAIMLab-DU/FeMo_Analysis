import unittest
from unittest import TestCase
import os
import pandas as pd
import numpy as np
from data.preprocess import DataPreprocessor


class TestPreprocess(TestCase):
    data_folder = 'tests/datafiles'
    input_folder = os.path.join(data_folder, 'input')
    output_folder = os.path.join(data_folder, 'output')

    data_preprocessor = DataPreprocessor(
        base_dir=data_folder
    )

    def test_load_data(self):

        for folder in os.listdir(self.input_folder):
            for file in os.listdir(os.path.join(self.input_folder, folder)):
                if file.endswith('.dat'):
                    loaded_data = self.data_preprocessor.load_data_file(os.path.join(self.input_folder, folder, file))

                    for k, v in loaded_data.items():
                        if isinstance(v, np.ndarray):
                            out_file = os.path.join(self.output_folder, 
                                                    folder, 
                                                    f"{k}.npz")
                            output_data = np.load(out_file)['arr_0']
                            np.testing.assert_allclose(
                                actual=v,  # your function output
                                desired=output_data,  # the expected output
                                rtol=1e-8,  # relative tolerance
                                atol=1e-12  # absolute tolerance
                            )
                        if isinstance(v, pd.DataFrame):
                            out_file = os.path.join(self.output_folder, 
                                                    folder, 
                                                    f"{k}.pkl")
                            output_data = pd.read_pickle(out_file)
                            pd.testing.assert_frame_equal(v, output_data)
    
    def test_preprocessed_data(self):

        for folder in os.listdir(self.input_folder):
            for file in os.listdir(os.path.join(self.input_folder, folder)):
                if file.endswith('.dat'):
                    loaded_data = self.data_preprocessor.load_data_file(os.path.join(self.input_folder, 
                                                                                     folder, 
                                                                                     file))
                    preprocessed_data = self.data_preprocessor.preprocess_data(loaded_data)

                    for k, v in preprocessed_data.items():
                        if isinstance(v, np.ndarray):
                            out_file = os.path.join(self.output_folder, 
                                                    folder, 
                                                    f"preprocessed-{k}.npz")
                            output_data = np.load(out_file)['arr_0']
                            np.testing.assert_allclose(
                                actual=v,  # your function output
                                desired=output_data,  # the expected output
                                rtol=1e-8,  # relative tolerance
                                atol=1e-12  # absolute tolerance
                            )
                        if isinstance(v, pd.DataFrame):
                            output_data = pd.read_pickle(os.path.join(self.output_folder, 
                                                                      folder, 
                                                                      f"preprocessed-{k}.pkl"))
                            pd.testing.assert_frame_equal(v, output_data)


if __name__ == '__main__':
    unittest.main()