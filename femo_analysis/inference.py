# TODO: implement functionality
import os
import sys
import yaml
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..', 'src')))
import argparse
import pandas as pd
from tqdm import tqdm
from logger import LOGGER
from keras.models import load_model
from data.dataset import FeMoDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasetDir", type=str, help="Directory containing inference dataset file")
    parser.add_argument("ckptFile", type=str, help="Path to model checkpoint file")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    parser.add_argument("--outfile", type=str, default="meta_info.csv", help="Output file containing meta info")
    args = parser.parse_args()

    return args


def main():
    LOGGER.info("Starting inference...")
    args = parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    LOGGER.info(f"Working directory {args.work_dir}")

    dataset: FeMoDataset = joblib.load(os.path.join(args.datasetDir, 'inference_dataset.pkl'))
    LOGGER.info(f"Loaded inference dataset: {os.path.abspath(args.datasetDir)}")

    ckpt_ext = os.path.basename(args.ckptFile).split('.')[-1]
    if ckpt_ext == '.h5':
        model = load_model(args.ckptFile)
    elif ckpt_ext == '.pkl':
        model = joblib.load(args.ckptFile)
    else:
        raise NotImplementedError(f"Model checkpoint with extension {ckpt_ext} not supported")
    
    for i in tqdm(range(len(dataset.data_manifest['items'])), desc="Inferencing files..."):
        item = dataset.data_manifest['items'][i]
        data_file_key = item.get('csvFileKey', None)
        if data_file_key is None:
            LOGGER.warning(f"{data_file_key = }")
            continue
        
        ...


if __name__ == "__main__":
    main()