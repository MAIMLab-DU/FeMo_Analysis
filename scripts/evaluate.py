import os
import yaml
import json
import pandas as pd
import argparse
from tqdm import tqdm
from femo.logger import LOGGER
from femo.eval.metrics import FeMoMetrics
from femo.data.dataset import FeMoDataset

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", type=str, required=True, help="Path to data manifest json file")
    parser.add_argument("--results-path", type=str, required=True, help="Path to file containing prediction results")
    parser.add_argument("--metadata-path", type=str, required=True, help="Path to file containing prediction metadata")
    parser.add_argument("--config-path", type=str, default=os.path.join(BASE_DIR, "..", "configs/dataset-cfg.yaml"), help="Path to config file")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to directory containing .dat and .csv files")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    parser.add_argument("--out-filename", type=str, default="performance.csv", help="Metrics output filename")
    parser.add_argument("--sagemaker", type=str, default=None, help="Sagemaker directory for results")
    args = parser.parse_args()

    return args


def main(args):
    LOGGER.info("Starting evaluation...")

    with open(args.config_path, 'r') as f:
        dataset_cfg = yaml.safe_load(f)

    results_df = pd.read_csv(args.results_path, index_col=False)
    LOGGER.info(f"Loaded results from {args.results_path}")

    dataset = FeMoDataset(
        base_dir=args.data_dir,
        data_manifest=args.data_manifest,
        pipeline_cfg=dataset_cfg.get('data_pipeline'),
        inference=False,
    )
    
    metrics_cfg = {
        key: dataset_cfg.get('data_pipeline')['segment'][key] for key in [
            'sensor_freq', 'sensation_freq', 'sensor_selection',
            'maternal_dilation_forward', 'maternal_dilation_backward',
            'fm_dilation'
        ]
    }
    metrics = FeMoMetrics(**metrics_cfg)
    overall_tpfp = {
        'true_positive': 0,
        'false_positive': 0,
        'true_negative': 0,
        'false_negative': 0
    }
    filewise_tpfp = {
        'filename': [],
        'true_positive': [],
        'false_positive': [],
        'true_negative': [],
        'false_negative': [],
        'sensation_calc': [],
        'sensation_label': [],
        'sensor_calc': [],
        'sensor_label': [],
        'ml_calc': [],
        'ml_label': []
    }

    for item in tqdm(dataset.data_manifest['items'], desc="Processing items", unit="item"):

        data_file_key = item.get('datFileKey', None)
        bucket=item.get('bucketName', None)
        data_filename = os.path.join(dataset.base_dir, data_file_key)
        data_success = dataset._download_from_s3(
            filename=data_filename,
            bucket=bucket,
            key=data_file_key
        )
        if not data_success:
            LOGGER.warning(f"Failed to download {data_filename} from {bucket = }, {data_file_key =}")
            continue

        pipeline_output = dataset.pipeline.process(
            filename=os.path.join(args.data_dir, data_file_key),
            outputs=['sensation_map', 'scheme_dict']
        )
        prediction_results = results_df[results_df['dat_file_key'] == data_file_key]

        tpfp_dict = metrics.calc_tpfp(
            sensation_map=pipeline_output['sensation_map'],
            scheme_dict=pipeline_output['scheme_dict'],
            pred_results=prediction_results,
        )
        filewise_tpfp['filename'].append(data_file_key)
        
        # TODO: save filewise tpfp
        for key in overall_tpfp.keys():
            value = tpfp_dict.get(key, 0)
            filewise_tpfp[key].append(value)
            overall_tpfp[key] += value

        # calc and label should be the same for all files
        filewise_tpfp['sensation_calc'].append(tpfp_dict['true_positive'] + tpfp_dict['false_negative'])
        filewise_tpfp['sensation_label'].append(tpfp_dict['num_maternal_sensed'])
        filewise_tpfp['sensor_calc'].append(tpfp_dict['true_positive'] + tpfp_dict['false_positive'] + tpfp_dict['true_negative'] + tpfp_dict['false_negative'])
        filewise_tpfp['sensor_label'].append(tpfp_dict['num_sensor_detections'])
        filewise_tpfp['ml_calc'].append(tpfp_dict['true_positive'] + tpfp_dict['false_positive'])
        filewise_tpfp['ml_label'].append(tpfp_dict['num_ml_detections'])
    
    outfile_dir = os.path.join(args.work_dir, "performance")
    os.makedirs(outfile_dir, exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, 'metrics'), exist_ok=True)

    perf_dict = metrics.calc_metrics(
        overall_tpfp
    )

    if args.sagemaker is None:
        perf_dict = {
            key: [val] for key, val in perf_dict.items()
        }
        perf_df = pd.DataFrame.from_dict(perf_dict)
        filewise_df = pd.DataFrame.from_dict(filewise_tpfp)

        perf_df.to_csv(os.path.join(outfile_dir, args.out_filename), index=False)
        filewise_df.to_csv(os.path.join(outfile_dir, f'filewise_{args.out_filename}'))

    else:
        perf_dict = {
            "classification_metrics": {
                key: {"value": val} for key, val in perf_dict.items()
            }
        }
        with open(os.path.join(outfile_dir, "evaluation.json"), 'w') as f:
            f.write(json.dumps(perf_dict, indent=2))

    metrics.save(os.path.join(args.work_dir, 'metrics'))
    LOGGER.info(f"Performance file saved as {os.path.abspath(outfile_dir)}")
    LOGGER.info(f"Metrics object saved as {os.path.abspath(os.path.join(args.work_dir, 'metrics'))}")


if __name__ == "__main__":
    args = parse_args()
    if args.sagemaker is not None:
        import tarfile
        def is_within_directory(directory, target):         
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)

            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        def safe_extract(tar_filepath, path="."):
            with tarfile.open(tar_filepath) as tar:
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tar.extractall(path)
        try:
            safe_extract(os.path.join(args.sagemaker, "output.tar.gz"),
                        path=args.sagemaker)
        except Exception:
            raise FileNotFoundError(f"No 'output.tar.gz' in {args.sagemaker}")

    main(args)