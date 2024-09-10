import json
from pathlib import Path
from typing import Union
from .pipeline import Pipeline


# TODO: add functionality
class FeMoDataset(object):

    @property
    def base_dir(self) -> Path:
        return self._base_dir
    
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
        self.pipeline = pipeline
        self._data_manifest_path = Path(data_manifest_path)
        self._data_manifest = None
        self.data_files = self._get_data_files(self.data_manifest)

    def _get_data_files(self, data_manifest) -> list:
        data_files = []
        for entry in data_manifest['data_files']:
            bucket = entry['bucketName']
            file = entry['objectKey']
            data_files.append(Path(self.base_dir, bucket, file))
        return data_files
