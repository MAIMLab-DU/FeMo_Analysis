import os
import sys
import joblib
import pandas as pd
import argparse
from tqdm import tqdm
from femo.logger import LOGGER
from femo.data.pipeline import Pipeline
from femo.data.transforms import SensorFusion
from femo.eval.metrics import FeMoMetrics
from femo.plot.plotter import FeMoPlotter
from femo.data.process import Processor
from femo.model.base import FeMoBaseClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, required=True, help="Path to data file")
    parser.add_argument("--model", type=str, required=True, help="Path to trained classifier file (.joblib)")
    parser.add_argument("--pipeline", type=str, required=True, help="Path to data pipeline object (.joblib)")
    parser.add_argument("--processor", type=str, required=True, help="Path to data processor object (.joblib)")
    parser.add_argument("--metrics", type=str, required=True, help="Path to evaluation metrics object (.joblib)")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    parser.add_argument("--outfile", type=str, default="meta_info.xlsx", help="Metrics output file")
    args = parser.parse_args()

    return args


def main(args):
    LOGGER.info("Starting inference...")

    if args.data_file.endswith('.dat'):
        # If it's a .dat file, store it in a list
        filenames = [args.data_file]
    elif args.data_file.endswith('.txt'):
        # If it's a .txt file, read each line and store filenames in a list
        with open(args.data_file, 'r') as file:
            filenames = [line.strip() for line in file if line.strip()]  # Strips newlines and empty lines
            filenames = [line for line in filenames if line.endswith('.dat')]
    else:
        raise ValueError("Unsupported file format. Please provide a .dat or .txt file.")
    LOGGER.info(f"Filenames: {filenames}")

    try:
        pipeline: Pipeline = joblib.load(args.pipeline)
        pipeline.inference = True
        processor: Processor = joblib.load(args.processor)
        classifier: type[FeMoBaseClassifier] = joblib.load(args.model)
        metrics: FeMoMetrics = joblib.load(args.metrics)
    except Exception as e:
        LOGGER.error(f"Error loading fitted objects: {e}")
        sys.exit(1)
        
    for data_filename in tqdm(filenames, desc=f"Peforming inference on {len(filenames)} files..."):
        pipeline_output = pipeline.process(
                filename=data_filename
            )
        
        X_extracted = pipeline_output['extracted_features']['features']

        X_norm_ranked = processor.predict(X_extracted)

        y_pred_labels = classifier.predict(X_norm_ranked)

        metainfo_dict, ml_map = metrics.calc_meta_info(
            filename=data_filename,
            y_pred=y_pred_labels,
            preprocessed_data=pipeline_output['preprocessed_data'],
            fm_dict=pipeline_output['fm_dict'],
            scheme_dict=pipeline_output['scheme_dict']
        )

        # Check if the file exists
        if os.path.exists(args.outfile):
            # If file exists, read it and append the new data
            df_existing = pd.read_excel(args.outfile)
            df_new = pd.DataFrame(metainfo_dict)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            # If file does not exist, create a new DataFrame with titles and new data
            df_combined = pd.DataFrame(metainfo_dict)

        # Write the combined DataFrame to the Excel file
        df_combined.to_excel(os.path.join(args.work_dir, args.outfile), index=False)

        LOGGER.info(f"Meta info saved to {os.path.join(args.work_dir, args.outfile)}")

        LOGGER.info("Plotting the results. It may take some time....")
        # Probably make this configurable
        plt_cfg = {
            'figsize': [16, 15],
            'x_unit': 'min'
        }
        plotter = FeMoPlotter()
        plotter.create_figure(
            figsize=plt_cfg['figsize']
        )
        i = 0
        for sensor_type in plotter.sensor_map.keys():
            for j in range(len(plotter.sensor_map[sensor_type])):
                sensor_name = f"{sensor_type}_{j+1}"
                plotter.plot_sensor_data(
                    axis_idx=i,
                    data=pipeline_output['preprocessed_data'][plotter.sensor_map[sensor_type][j]],
                    sensor_name=sensor_name,
                    x_unit=plt_cfg.get('x_unit', 'min')
                )
                i += 1

        # TODO: might be better to have this more configurable
        fusion_stage: SensorFusion = pipeline.stages[3]
        desired_scheme = fusion_stage.desired_scheme

        plotter.plot_detections(
            axis_idx=i,
            detection_map=pipeline_output['scheme_dict']['user_scheme'],
            det_type=f"At least {desired_scheme[1]} {desired_scheme[0]} Sensor Events",
            ylabel='Detection',
            xlabel='',
            x_unit=plt_cfg.get('x_unit', 'min')
        )
        plotter.plot_detections(
            axis_idx=i+1,
            detection_map=ml_map,
            det_type='Fetal movement',
            ylabel='Detection',
            xlabel=f"Time ({plt_cfg.get('x_unit', 'min')})",
            x_unit=plt_cfg.get('x_unit', 'min')
        )
        plotter.save_figure(os.path.join(args.work_dir, os.path.basename(data_filename).replace('.dat', '.png')))
        

if __name__ == "__main__":
    args = parse_args()
    main(args)
