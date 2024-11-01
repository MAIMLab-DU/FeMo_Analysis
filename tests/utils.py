import os
import numpy as np
import pandas as pd


def list_folders(directory):
    # List only folders (directories) in the specified directory
    folders = [
        f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))
    ]
    return folders


def compare_elements(key, actual, desired):
    if isinstance(actual, int) or isinstance(actual, float):
        assert actual == desired, f"{key} mismatched: {actual} != {desired}"

    if isinstance(actual, np.ndarray):            
        if actual.dtype == float:
            np.testing.assert_allclose(
                actual=actual, desired=desired, rtol=1e-4, atol=1,
                err_msg=f"{key} mismatched {np.argwhere(~np.isclose(actual, desired, rtol=1e-4, atol=1))}"
            )
        if actual.dtype == int or actual.dtype == bool:
            np.testing.assert_equal(
                actual=actual, desired=desired, err_msg=f"{key} mismatched"
            )

    if isinstance(actual, pd.DataFrame):
        pd.testing.assert_frame_equal(left=actual, right=desired, rtol=1e-4, atol=1)


def compare_dictionaries(actual_dict: dict, desired_dict: dict, keys: list = None):
    comparing_keys = keys if keys is not None else actual_dict.keys()
    for key in comparing_keys:
        if key not in desired_dict.keys():
            raise KeyError(f"'{key}' not in {desired_dict.keys()}")
        actual = actual_dict[key]
        desired = desired_dict[key]

        if isinstance(actual, dict):
            compare_dictionaries(actual, desired)

        if not isinstance(actual, list):
            compare_elements(key, actual, desired)
        else:
            assert len(actual) == len(desired), f"'{key}' {len(actual) = } != {len(desired) = }"
            for i in range(len(actual)):
                compare_elements(key, actual[i], desired[i])
        

        
