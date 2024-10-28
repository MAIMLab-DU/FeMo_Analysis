import os
import sys
import yaml
import joblib
import argparse
from femo.logger import LOGGER
from femo.model.base import FeMoBaseClassifier
from femo.model import CLASSIFIER_MAP




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasetDir", type=str, help="Directory containing train test pickle files")
    parser.add_argument("ckptName", type=str, help="Name of model checkpoint file")
    parser.add_argument("--tune", action='store_true', help="Tune hyperparameters before training")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    args = parser.parse_args()

    return args


def main():
    LOGGER.info("Starting training...")
    args = parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'train-cfg.yaml')
    with open(config_path, "r") as f:
        train_cfg = yaml.safe_load(f)

    LOGGER.info(f"Working directory {args.work_dir}")
    split_dataset = joblib.load(os.path.join(args.datasetDir, 'split_dataset.pkl'))

    train_data = split_dataset['train']
    test_data = split_dataset['test']
    LOGGER.info(f"Loaded train: {os.path.abspath(args.datasetDir)} " +
                f"test: {os.path.abspath(args.datasetDir)}")
    
    try:
        classifier_type = train_cfg.get('type')
        tune_params = train_cfg.get('tune_params', {})
        train_params = train_cfg.get('train_params', {})
        
        classifier: type[FeMoBaseClassifier] = CLASSIFIER_MAP[classifier_type]
        classifier_cfg = train_cfg.get('config', None)
        if classifier_cfg is None:
            classifier = classifier()
        else:
            classifier = classifier(config=classifier_cfg)
    except KeyError:
        LOGGER.error(f"Invalid classifier specified: {train_cfg.get('classifier')}")
        sys.exit(1)

    LOGGER.info(f"Training started with {classifier}")
    if args.tune:
        classifier.tune(
            train_data,
            test_data,
            **tune_params
        )
    classifier.fit(
        train_data,
        test_data,
        **train_params
    )
    try:
        classifier.save_model(
            model_filename=os.path.join(args.work_dir, args.ckptName),
            model_framework='keras' if classifier_type == 'neural-net' else 'sklearn'
        )
    except Exception:
        return
    LOGGER.info("Finished training")
    LOGGER.info(f"Model checkpoint saved to {os.path.join(args.work_dir, args.ckptName)}")
    results_path = os.path.join(args.work_dir, 'results.csv')
    classifier.result.compile_results(
        split_dict=split_dataset,
        filename=results_path
    )
    LOGGER.info(f"Results saved to {os.path.abspath(results_path)}")


if __name__ == "__main__":
    main()