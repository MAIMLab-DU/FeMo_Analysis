import os
import sys
import yaml
import argparse
import pandas as pd
from femo.logger import LOGGER
from femo.model.base import FeMoBaseClassifier
from femo.model import CLASSIFIER_MAP

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    
    # SageMaker-specific directories
    parser.add_argument("--train", type=str, default=os.getenv('SM_CHANNEL_TRAIN', ''), help="Path to dataset")
    parser.add_argument("--model-dir", type=str, default=os.getenv('SM_MODEL_DIR', ''), help="Model output directory")
    parser.add_argument("--output-data-dir", type=str, default=os.getenv('SM_OUTPUT_DATA_DIR', ''), help="Model output directory")

    # Additional arguments
    parser.add_argument("--ckpt-name", type=str, default=os.getenv('SM_CKPT_NAME', None), help="Name of model checkpoint file")
    parser.add_argument("--tune", action='store_true', default=os.getenv('SM_TUNE', False), help="Tune hyperparameters before training")
    parser.add_argument("--config-path", type=str, default=os.getenv('SM_TRAIN_CFG', os.path.join(BASE_DIR, "..", "configs/train-cfg.yaml")),
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
        train_cfg = yaml.safe_load(f)

    try:
        dataset = pd.read_csv(os.path.join(args.train, "dataset.csv"))
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
    
    if args.ckpt_name is None or args.ckpt_name == 'null':
        args.ckpt_name = "model.h5" if classifier_type == "neural-net" else "model.pkl"
    model_filename = os.path.join(args.model_dir, args.ckpt_name)
    try:
        classifier.save_model(
            model_filename=model_filename,
            model_framework='keras' if classifier_type == 'neural-net' else 'sklearn'
        )
        LOGGER.info(f"Model checkpoint saved to {os.path.abspath(model_filename)}")
    except Exception as e:
        LOGGER.error(f"Failed to save model: {e}")
        sys.exit(1)
    
    LOGGER.info("Finished training")

    results_path = os.path.join(args.output_data_dir, 'results')
    metadata_path = os.path.join(args.output_data_dir, 'metadata')
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(metadata_path, exist_ok=True)
    
    classifier.result.compile_results(
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
    main(args)
