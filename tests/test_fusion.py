import unittest
from unittest import TestCase
import os
import numpy as np
from collections import defaultdict
from data.fusion import SensorFusion


class TestFusion(TestCase):
    data_folder = 'tests/datafiles'
    input_folder = os.path.join(data_folder, 'input')
    output_folder = os.path.join(data_folder, 'output')

    data_fusion = SensorFusion(
        base_dir=data_folder
    )

    def test_user_scheme(self):
        
        allowed_files = ['fm_map.npz', 'fm_segmented.npz', 'fm_threshold.npz']
        for folder in os.listdir(self.input_folder):
            fm_dict = defaultdict()

            for file in os.listdir(os.path.join(self.input_folder, folder)):
                if file in allowed_files:
                    key = file.split('.npz')[0]
                    fm_dict[key] = np.load(os.path.join(self.input_folder, folder, file))['arr_0']
                    
            labeled_user_scheme, in_num_labels = self.data_fusion.get_labeled_user_scheme(fm_dict)
            out_file = os.path.join(self.output_folder, 
                                    folder, 
                                    f"labeled_user_scheme.npz")
            output_data = np.load(out_file)['arr_0']
            out_num_labels = len(np.unique(output_data)) - 1  # -1 to exclude background

            assert in_num_labels == out_num_labels, f"{in_num_labels} != {out_num_labels}"
            np.testing.assert_allclose(
                actual=labeled_user_scheme,
                desired=output_data, 
                rtol=1e-5, 
                atol=200
            )


if __name__ == '__main__':
    unittest.main()