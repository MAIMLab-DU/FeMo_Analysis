import os
import boto3
import json
import logging
import boto3.exceptions
import pandas as pd
from pathlib import Path
from typing import Union
from collections import defaultdict
from .pipeline import Pipeline


class DatasetBuilder(object):

    @property
    def logger(self):
        return logging.getLogger(__name__)

    @property
    def base_dir(self) -> Path:
        return self._base_dir
    
    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline
    
    @property
    def data_manifest(self) -> Path:
        if self._data_manifest is None:
            assert self._data_manifest_path.suffix == '.json', \
                "only json format is supported"
            self._data_manifest = json.load(self._data_manifest_path.open())
        return self._data_manifest
    
    @property
    def version(self) -> str:
        return self.data_manifest.get('version', None)
    
    def __init__(self,
                 base_dir: Union[Path, str],
                 data_manifest_path: Union[Path, str],
                 pipeline: Pipeline
                ) -> None:
        
        self._base_dir = Path(base_dir)
        self._pipeline = pipeline
        self._data_manifest_path = Path(data_manifest_path)
        self._data_manifest = None
        self.features_df = pd.DataFrame([])
        self.map = defaultdict()
    
    def __len__(self):
        return self.features_df.shape[0]
    
    def __repr__(self):
        return f"{self.__class__.__name__}(features_df={repr(self.features_df)})"
    
    def __getitem__(self, key):
        """Get the rows of features for the given key (filename)"""
        return self.features_df.iloc[self.map[key][0]:self.map[key][1], :]

    def _download_file(self, filename, bucket=None, key=None):

        if bucket is not None and key is not None:
            Path(f"{self._base_dir}/data").mkdir(parents=True, exist_ok=True)
            if not os.path.exists(filename):
                try:
                    s3 = boto3.resource("s3")
                    s3.Bucket(bucket).download_file(key, filename)
                    os.unlink(filename)
                    self.logger.info(f"Downloaded {filename} from {bucket = }, {key = }")
                except boto3.exceptions.ResourceNotExistsError:
                    return False
            return True
        
    def _save_features(self, filename, data: dict) -> pd.DataFrame:
        """Saves the extracted features (and labels) to a .csv file"""
        
        features = data.get('features')
        columns = data.get('columns')
        labels = data.get('labels')

        if features is None:
            raise ValueError("Missing features")
        
        features_df = pd.DataFrame(features, columns=columns)        
        if labels is not None:
            features_df['labels'] = labels
        features_df.to_csv(filename, header=columns is not None, index=False)
        
        return features_df
    
    def build(self):

        for item in self.data_manifest['items']:
            start_idx = len(self.features_df)

            bucket = item.get('bucketName', None)
            data_file_key = item.get('datFileKey', None)
            feat_file_key = item.get('csvFileKey', None)

            data_filename = os.path.join(self.base_dir, 'data', os.path.basename(data_file_key))
            data_success = self._download_file(
                filename=data_filename,
                bucket=bucket,
                key=data_file_key
            )
            if not data_success:
                self.logger.warning(f"Failed to download {data_filename} from {bucket = }, {data_file_key =}")
                continue

            feat_filename = os.path.join(self.base_dir, 'data', os.path.basename(feat_file_key))
            feat_success = self._download_file(
                filename=feat_filename,
                bucket=bucket,
                key=feat_file_key
            )
            if feat_success:
                current_features = pd.read_csv(feat_filename, index_col=False)
            else:
                extracted_features = self.pipeline.process(filename=data_filename)['extracted_features']
                try:
                    current_features = self._save_features(filename=feat_filename, data=extracted_features)
                except ValueError:
                    continue
                
            self.features_df = pd.concat([self.features_df, current_features], axis=0)   

            # Create mapping to get features give a filename from the dataset
            end_idx = len(self.features_df)
            map_key = os.path.basename(feat_file_key).split('.')[0]
            self.map[map_key] = (start_idx, end_idx)
        
        self.logger.info("FeMoDataset process completed.")
        return self.features_df


class DatasetProcessor:
    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def __init__(self, input_data: pd.DataFrame) -> None:
        self._input_data = input_data

    def process(self):
        ...
        
