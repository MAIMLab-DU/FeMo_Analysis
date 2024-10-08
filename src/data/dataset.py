import os
import boto3
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from logger import LOGGER
from typing import Literal
from botocore.exceptions import (
    ClientError,
    NoCredentialsError
)
from pathlib import Path
from typing import Union
from collections import defaultdict
from ._utils import gen_hash, stratified_kfold
from .pipeline import Pipeline
from .ranking import FeatureRanker


class FeMoDataset:

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
                 inference: bool,
                 pipeline_cfg: dict
                ) -> None:
        
        self._base_dir = Path(base_dir)
        self._pipeline = Pipeline(
            inference=inference,
            cfg=pipeline_cfg
        )
        self._data_manifest_path = Path(data_manifest_path)
        self._data_manifest = None
        self.features_df = pd.DataFrame([])
        self.map = defaultdict()
    
    def __len__(self):
        return self.features_df.shape[0]
    
    def __repr__(self):
        return f"{self.__class__.__name__}(features_df={repr(self.features_df)})"
    
    def __getitem__(self, filename: str):
        """Get the rows of features for the given key (filename)"""
        filename_hash = gen_hash(filename)
        return self.features_df[self.features_df['filename_hash'] == filename_hash]
    
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
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        if bucket and key:
            download_dir = Path(self._base_dir)
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
        
    def _save_features(self, filename, data: dict, key: str | None = None) -> pd.DataFrame:
        """Saves the extracted features (and labels) to a .csv file"""
        
        features = data.get('features')
        columns = data.get('columns')
        labels = data.get('labels')
        det_indices = data.get('det_indices')

        if features is None:
            raise ValueError("Missing features")
        
        features_df = pd.DataFrame(features, columns=columns)

        # Add optional columns based on existence
        if key is not None:
            features_df['filename_hash'] = key
        if det_indices is not None:
            features_df['det_indices'] = det_indices
        if labels is not None:
            features_df['labels'] = labels

        # Ensure the columns order is: filename_hash, det_indices, labels
        features_df.to_csv(filename, header=columns is not None, index=False)
        
        return features_df
    
    def build(self, force_extract: bool = False):

        for item in tqdm(self.data_manifest['items'], desc="Processing items", unit="item"):
            start_idx = len(self.features_df)

            bucket = item.get('bucketName', None)
            data_file_key = item.get('datFileKey', None)
            feat_file_key = item.get('csvFileKey', None)
            map_key = gen_hash(feat_file_key)
            if map_key in self.map.keys():
                continue

            data_filename = os.path.join(self.base_dir, data_file_key)
            data_success = self._download_from_s3(
                filename=data_filename,
                bucket=bucket,
                key=data_file_key
            )
            if not data_success:
                self.logger.warning(f"Failed to download {data_filename} from {bucket = }, {data_file_key =}")
                continue

            feat_filename = os.path.join(self.base_dir, feat_file_key)
            feat_success = self._download_from_s3(
                filename=feat_filename,
                bucket=bucket,
                key=feat_file_key
            )
            if feat_success and not force_extract:
                current_features = pd.read_csv(feat_filename, index_col=False)
            else:
                extracted_features = self.pipeline.process(filename=data_filename)['extracted_features']
                current_features = self._save_features(filename=feat_filename,
                                                       data=extracted_features,
                                                       key=map_key)
                try:
                    self._upload_to_s3(
                        filename=feat_filename,
                        bucket=bucket,
                        key=feat_file_key
                    )
                except Exception as e:
                    self.logger.warning(e)
                    pass

            self.features_df = pd.concat([self.features_df, current_features], axis=0, ignore_index=True)

            # Create mapping to get features give a filename from the dataset
            end_idx = len(self.features_df)
            self.map[map_key] = (start_idx, end_idx)
        
        self.logger.info("FeMoDataset process completed.")
        return self.features_df


class DataProcessor:
    @property
    def logger(self):
        return LOGGER
        
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
        # Avoid division by zero by adding a small epsilon
        norm_feats = norm_feats / dev

        return norm_feats[:, ~np.any(np.isnan(norm_feats), axis=0)]
    
    def split_data(self,
                    data: np.ndarray,
                    strategy: Literal['holdout', 'kfold'] = 'holdout',
                    custom_ratio: int|None = None,
                    num_folds: int = 5):

        X, y = data[:, :-1], data[:, -1]
        train, test = [], []
        split_data = defaultdict()

        if strategy == 'holdout':
            raise NotImplementedError("TODO")

        # TODO: no mod from original implementation
        elif strategy == 'kfold':
            X_TPD_norm = X[y == 1]
            X_FPD_norm = X[y == 0]

            folds_dict = stratified_kfold(X_TPD_norm, X_FPD_norm, custom_ratio, num_folds)
            for k in range(num_folds):
                # Combine the k-th fold's X and Y to form the test set
                test.append(np.concatenate([folds_dict['X_K_fold'][k], folds_dict['Y_K_fold'][k][:, np.newaxis]], axis=1))
                
                # Stack all folds except the k-th fold to form the training set
                X_train_current = np.vstack([folds_dict['X_K_fold'][i] for i in range(num_folds) if i != k])
                Y_train_current = np.concatenate([folds_dict['Y_K_fold'][i] for i in range(num_folds) if i != k])

                train.append(np.concatenate([X_train_current, Y_train_current[:, np.newaxis]], axis=1))

        split_data['train'] = train
        split_data['test'] = test
        for key, val in folds_dict.items():
            if key not in ('X_K_fold', 'Y_K_fold'):
                split_data[key] = val
        
        return split_data


    def process(self, input_data: pd.DataFrame, indices_filename: str|None = None):
        self.logger.debug("Processing features...")
        X_norm = self._normalize_features(input_data.drop(['labels', 'det_indices', 'filename_hash'],
                                                          axis=1, errors='ignore').to_numpy())
        y_pre = input_data.get('labels').to_numpy(dtype=int)
        det_indices = input_data.get('det_indices').to_numpy(dtype=int)
        filename_hash = input_data.get('filename_hash').to_numpy(dtype=int)
        
        # Check if top feature indices are present
        if indices_filename is None:
            top_feat_indices = self._feature_ranker.fit(X_norm, y_pre,
                                                        func=self._feature_ranker.ensemble_ranking)
        else:
            if not os.path.exists(indices_filename):
                top_feat_indices = self._feature_ranker.fit(X_norm, y_pre,
                                                            func=self._feature_ranker.ensemble_ranking)
            else:
                top_feat_indices = joblib.load(indices_filename)
                self.logger.info(f"Top features loaded from {indices_filename}")
        if indices_filename is not None:
            joblib.dump(top_feat_indices, indices_filename, compress=True)
            self.logger.info(f"Top features saved to {indices_filename}")

        X_norm = X_norm[:, top_feat_indices]

        # -3, -2, -1 are 'filename_hash', 'det_indices' and 'labels', respectively
        return np.concatenate([X_norm, filename_hash[:, np.newaxis],
                               det_indices[:, np.newaxis], y_pre[:, np.newaxis]], axis=1)
    



        
