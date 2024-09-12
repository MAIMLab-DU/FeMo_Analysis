import os
import boto3
import json
import logging
import numpy as np
import pandas as pd
from botocore.exceptions import (
    ClientError,
    NoCredentialsError
)
from pathlib import Path
from typing import Union
from collections import defaultdict
from .pipeline import Pipeline
from .ranking import FeatureRanker


class FeMoDataset:

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
    
    def _upload_to_s3(self, filename, bucket=None, key=None):

        s3 = boto3.client('s3')

        try:
            s3.upload_file(filename, bucket, key)
            print(f"File '{filename}' uploaded successfully to '{bucket}/{key}'.")

        except FileNotFoundError:
            print(f"File '{filename}' not found.")
        except NoCredentialsError:
            print("AWS credentials not available.")
        except ClientError as e:
            print(f"Error occurred while uploading to S3: {e}")

    def _download_from_s3(self, filename, bucket=None, key=None):
        
        if bucket and key:
            download_dir = Path(self._base_dir) / "data"
            download_dir.mkdir(parents=True, exist_ok=True)
            
            if not os.path.exists(filename):
                s3 = boto3.resource("s3")
                try:
                    s3.Bucket(bucket).download_file(key, filename)
                    self.logger.info(f"Downloaded {filename} from bucket={bucket}, key={key}")
                except ClientError as e:
                    self.logger.error(f"Failed to download {filename} from S3: {e}")
                    return False
                except Exception as e:
                    self.logger.error(f"Unexpected error while downloading {filename}: {e}")
                    return False
                
                return True
            else:
                self.logger.info(f"File {filename} already exists locally.")
                return True
        
        self.logger.error("Bucket or key not provided.")
        return False
        
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
            map_key = os.path.basename(feat_file_key).split('.')[0]
            if map_key in self.map.keys():
                continue

            data_filename = os.path.join(self.base_dir, 'data', os.path.basename(data_file_key))
            data_success = self._download_from_s3(
                filename=data_filename,
                bucket=bucket,
                key=data_file_key
            )
            if not data_success:
                self.logger.warning(f"Failed to download {data_filename} from {bucket = }, {data_file_key =}")
                continue

            feat_filename = os.path.join(self.base_dir, 'data', os.path.basename(feat_file_key))
            feat_success = self._download_from_s3(
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
                    self._upload_to_s3(
                        filename=feat_filename,
                        bucket=bucket,
                        key=feat_file_key
                    )
                except Exception as e:
                    self.logger.error(f"Error {e}")
                    continue
                
            self.features_df = pd.concat([self.features_df, current_features], axis=0)   

            # Create mapping to get features give a filename from the dataset
            end_idx = len(self.features_df)
            self.map[map_key] = (start_idx, end_idx)
        
        self.logger.info("FeMoDataset process completed.")
        return self.features_df


class DataProcessor:
    @property
    def logger(self):
        return logging.getLogger(__name__)
    
    @property
    def input_data(self) -> pd.DataFrame:
        return self._input_data
    
    @property
    def feat_rank_cfg(self) -> dict:
        return self._feat_rank_cfg     

    def __init__(self,
                 feat_rank_cfg: dict = None) -> None:
        
        self._feat_rank_cfg = feat_rank_cfg
        self._feature_ranker = FeatureRanker(**feat_rank_cfg) if feat_rank_cfg else FeatureRanker()

    def _normalize_features(self, data: np.ndarray):
        mu = np.mean(data, axis=0)
        norm_feats = data - mu
        dev = np.max(norm_feats, axis=0) - np.min(norm_feats, axis=0)
        norm_feats = norm_feats / dev

        return norm_feats[:, ~np.any(np.isnan(norm_feats), axis=0)]
    
    def train_test_split(self, data: np.ndarray,
                         num_folds: int,
                         tpd_fpd_ratio: float | None = None):
        
        X_tpd = data


    def process(self, input_data: pd.DataFrame):
        self.logger.debug("Processing features...")
        X_norm = self._normalize_features(input_data.drop('labels', axis=1, errors='ignore').to_numpy())
        y_pre = input_data.get('labels')
        
        top_feat_indices = self._feature_ranker.fit(X_norm, y_pre,
                                                    func=self._feature_ranker.ensemble_ranking)
        X_norm = X_norm[:, top_feat_indices]

        return np.concatenate([X_norm, y_pre[:, np.newaxis]], axis=1)
    



        
