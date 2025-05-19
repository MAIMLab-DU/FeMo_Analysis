import os
import json
import boto3
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Union
from collections import defaultdict
from botocore.exceptions import ClientError, NoCredentialsError
from ..logger import LOGGER
from .utils import gen_hash
from .pipeline import Pipeline


class FeMoDataset:

    def __init__(self,
                 base_dir: Union[str, Path],
                 data_manifest: Union[str, dict],
                 inference: bool,
                 pipeline_cfg: dict) -> None:

        self._base_dir = Path(base_dir).resolve()
        self._pipeline = Pipeline(inference=inference, cfg=pipeline_cfg)

        if isinstance(data_manifest, str):
            self._data_manifest_path = Path(data_manifest).resolve()
            self._data_manifest = None
        elif isinstance(data_manifest, dict):
            self._data_manifest_path = None
            self._data_manifest = data_manifest
        else:
            raise ValueError("data_manifest must be either a JSON path or a dict")

        self.inference = inference

    @property
    def logger(self):
        return LOGGER

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    @property
    def data_manifest(self) -> dict:
        if self._data_manifest is None:
            if not self._data_manifest_path.exists():
                raise FileNotFoundError(f"Manifest file not found: {self._data_manifest_path}")
            with open(self._data_manifest_path) as f:
                self._data_manifest = json.load(f)
        return self._data_manifest

    @property
    def version(self) -> str:
        return self.data_manifest.get('version')
    
    def _resolve_path(self, key: str) -> Path:
        path = Path(key)
        return path if path.is_absolute() else self.base_dir / path

    def _upload_to_s3(self, filename: str, bucket: str, key: str) -> None:
        s3 = boto3.client('s3')
        try:
            s3.upload_file(str(filename), bucket, key)
            self.logger.info(f"Uploaded to s3://{bucket}/{key}")
        except (FileNotFoundError, NoCredentialsError, ClientError) as e:
            self.logger.warning(f"Upload failed: {e}")

    def _download_from_s3(self, filename: str, bucket: str, key: str) -> bool:
        if not bucket or not key:
            self.logger.error("Missing bucket or key for S3 download.")
            return False

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if os.path.exists(filename):
            self.logger.info(f"File already exists locally: {filename}")
            return True

        s3 = boto3.resource("s3")
        try:
            s3.Bucket(bucket).download_file(key, filename)
            self.logger.info(f"Downloaded {filename} from s3://{bucket}/{key}")
            return True
        except ClientError as e:
            self.logger.warning(f"S3 download failed: {e}")
            return False

    def _save_features(self, filename: str, data: dict, key: str = None) -> pd.DataFrame:
        features = data.get('features')
        columns = data.get('columns')
        labels = data.get('labels')
        det_indices = data.get('det_indices')

        if features is None:
            raise ValueError("No features in extracted data.")

        features_df = pd.DataFrame(features, columns=columns)
        features_df['filename_hash'] = key
        features_df['det_indices'] = det_indices
        features_df['labels'] = labels

        features_df.to_csv(filename, index=False)
        return features_df

    def build(self, force_extract: bool = False, skip_upload: bool = False) -> dict[str, pd.DataFrame]:
        features_dict = defaultdict(pd.DataFrame)

        for item in tqdm(self.data_manifest.get('items', []), desc="Processing items", unit="item"):
            bucket = item.get('bucketName')
            data_file_key = item.get('datFileKey')
            feat_file_key = item.get('csvFileKey', item['datFileKey'].replace('.dat', '.csv'))
            remove_zeros = item.get('removeZeros', False)


            if not data_file_key or not feat_file_key:
                self.logger.warning("Skipping item due to missing datFileKey or csvFileKey.")
                continue

            map_key = gen_hash(data_file_key)
            data_filename = self._resolve_path(data_file_key)

            if not self._download_from_s3(str(data_filename), bucket, data_file_key):
                self.logger.warning(f"Skipping item due to missing data: {data_filename}")
                continue

            try:
                extracted, _ = self.pipeline.extract_features_batch(str(data_filename))
            except Exception as e:
                self.logger.error(f"Feature extraction failed for {data_filename}: {e}")
                continue

            for feature_set, feat_data in extracted.items():
                feat_filename = self.base_dir / feat_file_key.replace('.csv', f'-{feature_set}.csv')

                if feat_filename.exists() and not force_extract:
                    current_features = pd.read_csv(feat_filename)
                else:
                    current_features = self._save_features(str(feat_filename), feat_data, map_key)

                    if not skip_upload:
                        try:
                            self._upload_to_s3(str(feat_filename), bucket, feat_filename.name)
                        except Exception as e:
                            self.logger.warning(f"Upload skipped due to error: {e}")

                if(remove_zeros):
                    current_features = current_features[current_features["labels"] == 1]

                features_dict[feature_set] = pd.concat(
                    [features_dict[feature_set], current_features], axis=0
                )

        if not any(len(df) for df in features_dict.values()):
            msg = (
                "Feature extraction failed: no features were collected from any data item. "
                "Check if the dataset is empty, data downloads failed, or processing errors occurred."
            )
            self.logger.error(msg)
            raise RuntimeError(msg)

        self.logger.info("FeMoDataset build completed.")
        return dict(features_dict)
