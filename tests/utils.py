import os
import numpy as np
import pandas as pd


def list_folders(directory):
    # List only folders (directories) in the specified directory
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    return folders


def test_dictionaries(actual_dict: dict, desired_dict: dict):

    for key in actual_dict.keys():
        actual = actual_dict[key]
        desired = desired_dict[key]

        if isinstance(actual, int) or isinstance(actual, float):
            assert actual == desired, f"{actual =} != {desired = }"
        
        if isinstance(actual, np.ndarray):
            if actual.dtype == float:
                np.testing.assert_allclose(
                    actual=actual,
                    desired=desired,
                    rtol=1e-4,
                    atol=1
                )
            if actual.dtype == int:
                np.testing.assert_equal(
                    actual=actual,
                    desired=desired,
                    err_msg=f"{key} mismatched"
                )
        
        if isinstance(actual, pd.DataFrame):
            pd.testing.assert_frame_equal(
                left=actual,
                right=desired,
                rtol=1e-4,
                atol=1
            )