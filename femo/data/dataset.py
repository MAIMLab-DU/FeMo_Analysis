import os
import boto3
import json
import pandas as pd
from tqdm import tqdm
from ..logger import LOGGER
from botocore.exceptions import (
    ClientError,
    NoCredentialsError
)
from pathlib import Path
from typing import Union
from collections import defaultdict
from .utils import gen_hash
from .pipeline import Pipeline


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
    def feat_rank_cfg(self) -> dict:
        return self._feat_rank_cfg  
    
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
                 data_manifest: Union[str, dict],
                 inference: bool,
                 pipeline_cfg: dict,
                ) -> None:
        
        self._base_dir = Path(base_dir)
        self._pipeline = Pipeline(
            inference=inference,
            cfg=pipeline_cfg
        )
        if isinstance(data_manifest, str):
            self._data_manifest_path = Path(data_manifest)
            self._data_manifest = None
        elif isinstance(data_manifest, dict):
            self._data_manifest = data_manifest
        else:
            raise ValueError("data_manifest must be either a JSON path or a dict")
        
        self.inference = inference
    
    # def __len__(self):
    #     return self.features_df.shape[0]
    
    # def __repr__(self):
    #     return f"{self.__class__.__name__}(features_df={repr(self.features_df)})"
    
    # def __getitem__(self, filename: str):
    #     """Get the rows of features for the given key (filename)"""
    #     filename_hash = gen_hash(filename)
    #     return self.features_df[self.features_df['filename_hash'] == filename_hash]
    
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
        features_df['filename_hash'] = key
        features_df['det_indices'] = det_indices
        features_df['labels'] = labels

        # Ensure the columns order is: filename_hash, det_indices, labels
        features_df.to_csv(filename, header=columns is not None, index=False)
        
        return features_df
    
    def build(self, force_extract: bool = False):

        features_dict = {
            key: pd.DataFrame([]) for key in self.pipeline.stages[5].feature_sets
        }

        for item in tqdm(self.data_manifest['items'], desc="Processing items", unit="item"):

            bucket = item.get('bucketName', None)
            data_file_key = item.get('datFileKey', None)
            feat_file_key = item.get('csvFileKey', None)
            map_key = gen_hash(data_file_key)

            data_filename = os.path.join(self.base_dir, data_file_key)
            data_success = self._download_from_s3(
                filename=data_filename,
                bucket=bucket,
                key=data_file_key
            )
            if not data_success:
                self.logger.warning(f"Failed to download {data_filename} from {bucket = }, {data_file_key =}")
                continue

            for feature_set in self.pipeline.stages[5].feature_sets:
                feat_filename = os.path.join(self.base_dir, feat_file_key.replace('.csv', f'-{feature_set}.csv'))
                feat_success = self._download_from_s3(
                    filename=feat_filename,
                    bucket=bucket,
                    key=os.path.basename(feat_filename)
                )
                if feat_success and not force_extract:
                    current_features = pd.read_csv(feat_filename, index_col=False)
                else:
                    extracted_features: dict = self.pipeline.process(filename=data_filename, feature_set=feature_set)['extracted_features']
                    current_features = self._save_features(filename=feat_filename, 
                                                           data=extracted_features, 
                                                           key=map_key)
                    try:
                        self._upload_to_s3(
                            filename=feat_filename,
                            bucket=bucket,
                            key=os.path.basename(feat_filename)
                        )
                    except Exception as e:
                        self.logger.warning(e)
                        pass
                features_dict[feature_set] = pd.concat([features_dict[feature_set], current_features], axis=0)
        
        self.logger.info(f"FeMoDataset process completed.")
        return features_dict
