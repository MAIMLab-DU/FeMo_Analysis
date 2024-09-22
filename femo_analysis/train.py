# TODO: implement functionality
import os
import sys
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..', 'src')))
import argparse
from logger import LOGGER
from model.base import FeMoBaseClassifier
from model import CLASSIFIER_MAP




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset-dir", type=str, help="Directory containing train test pickle files")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'train-cfg.yaml')
    with open(config_path, "r") as f:
        train_cfg = yaml.safe_load(f)

    LOGGER.info(f"Training started with {train_cfg = }")
    LOGGER.info(f"Working directory {args.work_dir}")
    
    try:
        classifier_type = train_cfg.get('classifier')['type']
        
        classifier: type[FeMoBaseClassifier] = CLASSIFIER_MAP[classifier_type]
        classifier = classifier(**train_cfg.get('config'))
    except KeyError:
        LOGGER.error(f"Invalid classifier specified: {train_cfg.get('classifier')}")
        sys.exit(1)

    classifier.search()
    


if __name__ == "__main__":
    main()