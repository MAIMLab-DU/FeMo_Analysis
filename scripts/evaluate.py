import os
import json
import yaml
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path
from femo.logger import LOGGER
from femo.eval.metrics import FeMoMetrics
from femo.data.pipeline import Pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", type=str, required=True, help="Path to data manifest json file")
    parser.add_argument("--results-path", type=str, required=True, help="Directory containing prediction results")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to directory containing .dat and .csv files")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    parser.add_argument("--outfile", type=str, default="performance.csv", help="Metrics output file")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
    config_files = ['dataset-cfg.yaml']
    [dataset_cfg] = [yaml.safe_load(open(os.path.join(config_dir, cfg), 'r')) for cfg in config_files]

    with open(os.path.join(os.path.dirname(args.results_path), 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    results_df = pd.read_csv(args.results_path, index_col=False)

    overall_tpd_pred = results_df.get('predictions').to_numpy()[0: metadata['num_tpd']]
    overall_fpd_pred = results_df.get('predictions').to_numpy()[metadata['num_tpd']:]
    matching_index_tpd = 0
    matching_index_fpd = 0
    LOGGER.info(f"Loaded results from {args.results_path}")

    pipeline = Pipeline(
            inference=False,
            cfg=dataset_cfg.get('data_pipeline')
        )
    metrics_calculator = FeMoMetrics()
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
        'num_tpd': [],
        'num_fpd': []
    }

    data_manifest = json.load(Path(args.data_manifest).open())
    for item in tqdm(data_manifest['items'], desc="Processing items", unit="item"):
        data_file_key = item.get('datFileKey', None)
        if data_file_key is None:
            LOGGER.warning(f"{data_file_key = }")
            continue

        pipeline_output = pipeline.process(
            filename=os.path.join(args.data_dir, data_file_key),
            outputs=['preprocessed_data', 'imu_map', 'sensation_map', 'scheme_dict']

        )
        tpfp_dict = metrics_calculator.calc_tpfp(
            preprocessed_data=pipeline_output['preprocessed_data'],
            imu_map=pipeline_output['imu_map'],
            sensation_map=pipeline_output['sensation_map'],
            scheme_dict=pipeline_output['scheme_dict'],
            overall_tpd_pred=overall_tpd_pred,
            overall_fpd_pred=overall_fpd_pred,
            matching_index_tpd=matching_index_tpd,
            matching_index_fpd=matching_index_fpd
        )
        filewise_tpfp['filename'].append(data_file_key)
        
        # TODO: save filewise tpfp
        for key in overall_tpfp.keys():
            value = tpfp_dict.get(key, 0)
            filewise_tpfp[key].append(value)
            overall_tpfp[key] += value

        filewise_tpfp['sensation_calc'].append(tpfp_dict['true_positive'] + tpfp_dict['false_negative'])
        filewise_tpfp['sensation_label'].append(tpfp_dict['num_maternal_sensed'])
        filewise_tpfp['sensor_calc'].append(tpfp_dict['true_positive'] + tpfp_dict['false_positive'])
        filewise_tpfp['sensor_label'].append(tpfp_dict['num_sensor_sensed'])
        filewise_tpfp['num_tpd'].append(len(tpfp_dict['tpd_indices']))
        filewise_tpfp['num_fpd'].append(len(tpfp_dict['fpd_indices']))

        matching_index_tpd = tpfp_dict['matching_index_tpd']
        matching_index_fpd = tpfp_dict['matching_index_fpd']
        
    metrics_dict = metrics_calculator.calc_metrics(
        overall_tpfp
    )
    metrics_dict = {
        key: [val] for key, val in metrics_dict.items()
    }
    metrics_df = pd.DataFrame.from_dict(metrics_dict)
    filewise_df = pd.DataFrame.from_dict(filewise_tpfp)
    outfile = os.path.join(args.work_dir, args.outfile)
    metrics_df.to_csv(outfile, index=False)
    filewise_df.to_csv(os.path.join(args.work_dir, 'filewise_performance.csv'))
    LOGGER.info(f"Performance metrics saved as {os.path.abspath(outfile)}")


if __name__ == "__main__":
    main()