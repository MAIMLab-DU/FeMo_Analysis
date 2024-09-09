from pathlib import Path
import yaml
from typing import Union
from transforms import (
    DataLoader,
    DataPreprocessor,
    SensorFusion,
    DataSegmentor,
    DetectionExtractor,
    FeatureExtractor
)


class Pipeline(object):

    @property
    def transform_map(self):
        return {
            'load': DataLoader,
            'preprocess': DataPreprocessor,
            'fuse': SensorFusion,
            'segment': DataSegmentor,
            'extract_det': DetectionExtractor,
            'extract_feat': FeatureExtractor
        }

    @property
    def step_order(self):
        return ['load', 'preprocess',
                'fuse', 'segment',
                'extract_det', 'extract_feat']

    def __init__(self,
                 cfg_path: Union[str, Path],
                 inference: bool = False) -> None:

        self.inference = inference
        self.cfg_path = cfg_path
        self.pipeline_cfg = self._get_pipeline_cfg(cfg_path)
        self.pipeline = self._get_pipeline()

    @staticmethod
    def _get_pipeline_cfg(path=None):
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def _get_pipeline(self, pipeline_cfg: dict):
        pipeline = []
        for step in self.step_order:
            transform = self.transform_map.get(step)
            cfg = pipeline_cfg.get(step, None)
            pipeline.append(transform(**cfg))
        return pipeline

    def run_pipeline(self, filename):
        ...
