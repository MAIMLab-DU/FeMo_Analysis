import os
import yaml
import json
import argparse
import numpy as np
from tqdm import tqdm
from femo.data.dataset import FeMoDataset
from femo.data.transforms import DataSegmentor

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=str, help="Path to data manifest json file")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to directory containing .dat and .csv files")    
    parser.add_argument("--config-path", type=str, default=os.path.join(BASE_DIR, "..", "configs/dataset-cfg.yaml"), help="Path to config file")
    parser.add_argument("--nth-percentile", type=int, default=95, help="N-th percentile to calculate")
    args = parser.parse_args()

    return args


def calculate_segment_peaks(sensor_data: np.ndarray, sensation_data: np.ndarray, segmentor: DataSegmentor):

    M_event_index = np.where(sensation_data)[0]
    L1_prev, L2_prev = None, None
    sensor_sensation = []

    for j in range(len(M_event_index)):
        L = M_event_index[j]  # Sample no. corresponding to a maternal sensation
        L1 = L * round(segmentor.sensor_freq / segmentor.sensation_freq) - segmentor.extension_backward
        L2 = L * round(segmentor.sensor_freq / segmentor.sensation_freq) + segmentor.extension_forward
        L1 = max(L1, 0)  # Ensure L1 is not less than 0
        L2 = min(L2, len(sensation_data))  # Ensure L2 is within bounds

        # Check if current L1 overlaps with the previous L2
        if L1_prev is not None and L1 <= L2_prev:
            # Merge the current range with the previous one
            L1 = L1_prev  # Keep the start point of the previous range
            L2 = max(L2, L2_prev)  # Extend to the maximum end point
        else:
            # If no overlap, append previous sensation (only if there's a valid one)
            if L1_prev is not None:
                # np.max(np.abs( ))
                sensor_sensation.append(np.max(np.abs(sensor_data[L1_prev:L2_prev])))
        # Update L1_prev and L2_prev for the next iteration
        L1_prev, L2_prev = L1, L2

    # Append the last remaining range after the loop
    if L1_prev is not None:
        sensor_sensation.append(np.max(np.abs(sensor_data[L1_prev:L2_prev])))

    return sensor_sensation


def main(args):

    with open(args.config_path, 'r') as f:
        dataset_cfg = yaml.safe_load(f)

    with open(args.manifest, 'r') as f:
        data_manifest = json.load(f)
    
    print(len(data_manifest['items']))
    
    dataset = FeMoDataset(
        args.data_dir,
        data_manifest,
        False,
        dataset_cfg.get('data_pipeline')
    )

    sensor_sensation_peaks = {
        key: [] for key in dataset.pipeline.stages[0].sensors
    }

    for item in tqdm(data_manifest['items'], desc="Calculating segment peaks", unit="file"):
        bucket = item.get('bucketName', None)
        data_file_key = item.get('datFileKey', None)
        data_filename = os.path.join(args.data_dir, data_file_key)
        data_success = dataset._download_from_s3(
                filename=data_filename,
                bucket=bucket,
                key=data_file_key
            )
        if not data_success:
            dataset.logger.warning(f"Failed to download {data_filename} from {bucket = }, {data_file_key =}")
            continue

        loaded_data = dataset.pipeline.stages[0].transform(data_filename)
        preprocessed_data = dataset.pipeline.stages[1].transform(loaded_data)

        for sensor in dataset.pipeline.stages[0].sensors:
            sensor_data = preprocessed_data[sensor]
            sensation_data = preprocessed_data['sensation_data']
            peaks = calculate_segment_peaks(sensor_data, sensation_data, dataset.pipeline.stages[2])
            sensor_sensation_peaks[sensor].extend(peaks)

    data_manifest['percentiles'] = {
        'nth_percent': args.nth_percentile
    }
    for sensor, peaks in sensor_sensation_peaks.items():
        percentile = np.percentile(peaks, args.nth_percentile)
        dataset.logger.info(f"Total segments for {sensor} is {len(peaks)}")
        dataset.logger.info(f"{args.nth_percentile}-th percentile for {sensor} is {percentile}")
        data_manifest['percentiles'][sensor] = {
            'percentile': percentile,
            'total_segments': len(peaks)
        }

    with open(args.manifest, 'w') as f:
        json.dump(data_manifest, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
