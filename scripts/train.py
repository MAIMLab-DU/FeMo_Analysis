import os
import sys
import yaml
import argparse
import pandas as pd
from femo.logger import LOGGER
from femo.model.base import FeMoBaseClassifier
from femo.model import CLASSIFIER_MAP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset csv file")
    parser.add_argument("--ckpt-name", type=str, required=True, help="Name of model checkpoint file")
    parser.add_argument("--tune", action='store_true', help="Tune hyperparameters before training")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    args = parser.parse_args()

    return args


def main():
    LOGGER.info("Starting training...")
    args = parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    LOGGER.info(f"Working directory {args.work_dir}")

    config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
    config_files = ['train-cfg.yaml']
    [train_cfg] = [yaml.safe_load(open(os.path.join(config_dir, cfg), 'r')) for cfg in config_files]

    dataset = pd.read_csv(args.dataset_path)

    LOGGER.info(f"Loaded train: {os.path.abspath(args.dataset_path)} " +
                f"test: {os.path.abspath(args.dataset_path)}")
    
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

    split_cfg = train_cfg.get('split_config', {'strategy': 'kfold', 'num_folds': 5})
    LOGGER.info(f"Splitting dataset with {split_cfg}")
    train_data, test_data, metadata = classifier.split_data(
        dataset.to_numpy(),
        **split_cfg
    )

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
            model_filename=os.path.join(args.work_dir, args.ckpt_name),
            model_framework='keras' if classifier_type == 'neural-net' else 'sklearn'
        )
        LOGGER.info(f"Model checkpoint saved to {os.path.join(args.work_dir, args.ckpt_name)}")
    except Exception:
        LOGGER.error("Failed to save model to")
        pass
    LOGGER.info("Finished training")
    results_path = os.path.join(args.work_dir, 'results.csv')
    classifier.result.compile_results(
        metadata=metadata,
        filename=results_path
    )
    LOGGER.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()