import os
import boto3
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Union
from .pipeline import Pipeline


# TODO: add functionality
class FeMoDataset(object):

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
        return self.data_manifest['version']
    
    def __init__(self,
                 base_dir: Union[Path, str],
                 data_manifest_path: Union[Path, str],
                 pipeline: Pipeline
                ) -> None:
        
        self._base_dir = Path(base_dir)
        self._pipeline = pipeline
        self._data_manifest_path = Path(data_manifest_path)
        self._data_manifest = None

    def _download_file(self, filename, bucket=None, key=None):

        if bucket is not None and key is not None:
            Path(f"{self._base_dir}/data").mkdir(parents=True, exist_ok=True)
            if not os.path.exists(filename):
                s3 = boto3.resource("s3")
                s3.Bucket(bucket).download_file(key, filename)
                os.unlink(filename)
                self.logger.info(f"Downloaded {filename} from {bucket = }, {key = }")
            return
    
    def build(self):
        features_df = pd.DataFrame([])

        for item in self.data_manifest['items']:
            bucket = item.get('bucketName', None)
            data_file_key = item.get('datFileKey', None)
            feat_file_key = item.get('csvFileKey', None)

            data_filename = os.path.join(self.base_dir, 'data', os.path.basename(data_file_key))
            self._download_file(
                filename=data_filename,
                bucket=bucket,
                key=data_file_key
            )

            feat_filename = os.path.join(self.base_dir, 'data', os.path.basename(feat_file_key))
            self._download_file(
                filename=feat_filename,
                bucket=bucket,
                key=feat_file_key
            )
            current_features = pd.read_csv(feat_filename, header=None, index_col=False)
            ...



            
        
        self.logger.info("FeMoDataset build completed.")

