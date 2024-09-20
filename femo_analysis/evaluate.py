# TODO: implement functionality
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..', 'src')))
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results-dir", type=str, required=True, help="Directory containing prediction results")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    args = parser.parse_args()

    return args