import os
import sys
import json
import yaml
import joblib
import argparse
import pandas as pd
from femo.logger import LOGGER
from femo.model.base import FeMoBaseClassifier
from femo.model import CLASSIFIER_MAP

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    
    # SageMaker-specific directories
    parser.add_argument("--train", type=str, required=True, help="Path to dataset")
    parser.add_argument("--model-dir", type=str, required=True, help="Model output directory")
    parser.add_argument("--output-data-dir", type=str, required=True, help="Model output directory")

    # Additional arguments
    parser.add_argument("--tune", action='store_true', default=False, help="Tune hyperparameters before training")
    parser.add_argument("--config-path", type=str, default=os.path.join(BASE_DIR, "..", "configs/train-cfg.yaml"),
                        help="Path to config file")

    args = parser.parse_args()
    return args


def main(args):
    LOGGER.info("Starting training...")

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)
    LOGGER.info(f"Model directory: {args.model_dir}")
    LOGGER.info(f"Output data directory: {args.output_data_dir}")

    with open(args.config_path, 'r') as f:
        train_cfg = yaml.safe_load(f) if args.config_path.endswith('.yaml') else json.load(f)

    feature_sets = train_cfg.get('feature_sets', ['crafted'])

    dataset = pd.DataFrame([])
    try:
        for key in feature_sets:
            dataset = pd.concat([dataset, pd.read_csv(os.path.join(args.train, f"{key}_dataset.csv"))], axis=1)
    except Exception:
        raise ValueError(
            (
                "Preprocessed dataset not in {}.\n"
                + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                + "the data specification in S3 was incorrectly specified or the role specified\n"
                + "does not have permission to access the data."
            ).format(args.train, "train")
        )

    LOGGER.info(f"Loaded dataset from: {os.path.abspath(args.train)}")
    
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
    LOGGER.info(f"Splitting dataset with config: {split_cfg}")
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
        classifier.save_model(os.path.join(args.model_dir, 'model.joblib'))
        classifier.model = None
        joblib.dump(classifier, os.path.join(args.model_dir, 'classifier.joblib'), compress=False)
        LOGGER.info(f"Classifier and fitted model saved to: {os.path.abspath(os.path.join(args.model_dir))}")
    except Exception as e:
        LOGGER.error(f"Failed to save model: {e}")
        sys.exit(1)
    
    LOGGER.info("Finished training")

    results_path = os.path.join(args.output_data_dir, 'results')
    metadata_path = os.path.join(args.output_data_dir, 'metadata')
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(metadata_path, exist_ok=True)
    
    classifier.result.compile_results(
        threshold=train_cfg.get('threshold', 0.5),
        metadata=metadata,
        results_path=results_path,
        metadata_path=metadata_path
    )

    LOGGER.info(f"Predictions saved to {os.path.abspath(results_path)}")
    LOGGER.info(f"Metadata saved to {os.path.abspath(metadata_path)}")



if __name__ == "__main__":
    if sys.argv[1] == 'train':
        sys.argv.pop(1)    
    args = parse_args()
    if isinstance(args.tune, str):
        args.tune = args.tune.lower() == 'true'
    main(args)
