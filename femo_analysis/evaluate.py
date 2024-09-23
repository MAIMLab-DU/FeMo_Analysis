# TODO: implement functionality
import os
import sys
import json
import yaml
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..', 'src')))
import argparse
from tqdm import tqdm
from pathlib import Path
from logger import LOGGER
from data._utils import gen_hash
from eval.metrics import FeMoMetrics
from data.pipeline import Pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataManifest", type=str, help="Path to data manifest json file")
    parser.add_argument("resultsDir", type=str, help="Directory containing prediction results")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to directory containing .dat and .csv files")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    parser.add_argument("--outfile", type=str, default="performance.csv", help="Metrics output file")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'dataproc-cfg.yaml')
    with open(config_path, "r") as f:
        dataproc_cfg = yaml.safe_load(f)

    results_df = pd.read_csv(os.path.join(args.resultsDir, 'results.csv'), index_col=False)
    LOGGER.info(f"Loaded results from {os.path.abspath(args.resultsDir)}")

    pipeline = Pipeline(
            inference=False,
            cfg=dataproc_cfg.get('data_pipeline')
        )
    metrics_calculator = FeMoMetrics()
    overall_tpfp = {
        'true_positive': {key: 0 for key in metrics_calculator.sensors},
        'false_positive': {key: 0 for key in metrics_calculator.sensors},
        'true_negative': {key: 0 for key in metrics_calculator.sensors},
        'false_negative': {key: 0 for key in metrics_calculator.sensors}
    }

    data_manifest = json.load(Path(args.dataManifest).open())
    for item in tqdm(data_manifest['items'], desc="Processing items", unit="item"):
        data_file_key = item.get('datFileKey', None)
        if data_file_key is None:
            LOGGER.warning(f"{data_file_key = }")
            continue
        feat_file_key = item.get('csvFileKey', None)
        map_key = gen_hash(feat_file_key)

        pipeline_output = pipeline.process(
            filename=os.path.join(args.data_dir, data_file_key),
            outputs=['preprocessed_data', 'imu_map', 'sensation_map', 'scheme_dict']

        )
        tpfp_dict = metrics_calculator.calc_tpfp(
            preprocessed_data=pipeline_output['preprocessed_data'],
            imu_map=pipeline_output['imu_map'],
            sensation_map=pipeline_output['sensation_map'],
            scheme_dict=pipeline_output['scheme_dict'],
            results_df=results_df,
            filename_hash=map_key
        )
        
        # TODO: save filewise tpfp
        for key, value in tpfp_dict.items():
            for s, v in value.items():
                overall_tpfp[key][s] += v
        
    metrics_dict = metrics_calculator.calc_metrics(
        overall_tpfp
    )
    metrics_dict = {
        key: [val['sensor_1']] for key, val in metrics_dict.items()
    }
    metrics_df = pd.DataFrame.from_dict(metrics_dict)
    outfile = os.path.join(args.work_dir, args.outfile)
    metrics_df.to_csv(outfile, index=False)
    LOGGER.info(f"Performance metrics saved as {os.path.abspath(outfile)}")


if __name__ == "__main__":
    main()