import yaml
from pathlib import Path
from collections import defaultdict
from typing import Union
from .transforms import (
    DataLoader,
    DataPreprocessor,
    SensorFusion,
    DataSegmentor,
    DetectionExtractor,
    FeatureExtractor
)


class Pipeline(object):

    def __init__(self,
                 cfg_path: Union[str, Path],
                 inference: bool = False) -> None:

        self.inference = inference
        self.cfg_path = cfg_path
        self.pipeline_cfg = self._get_pipeline_cfg(cfg_path)
        self.stages = self._get_stages(
            pipeline_cfg=self.pipeline_cfg
        )

    @staticmethod
    def _get_pipeline_cfg(path=None):
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def _get_stages(self, pipeline_cfg: dict):
        stages = defaultdict()

        stages[0] = DataLoader(**pipeline_cfg.get('load', {}))
        stages[1] = DataPreprocessor(**pipeline_cfg.get('preprocess', {}))
        stages[2] = DataSegmentor(**pipeline_cfg.get('segment', {}))
        stages[3] = SensorFusion(**pipeline_cfg.get('fusion', {}))
        stages[4] = DetectionExtractor(**pipeline_cfg.get('extract_det', {}))
        stages[5] = FeatureExtractor(**pipeline_cfg.get('extract_feat', {}))

        return stages

    def process(self, filename):
        
        # Step-0: Load data
        loaded_data = self.stages[0](filename=filename)

        # Step-1: Preprocess data (filter and trimming)
        preprocessed_data = self.stages[1](loaded_data=loaded_data)

        # Step-2: Get imu_map, fm_dict (fm sensors), and sensation_dict (button)
        imu_map = self.stages[2](
            map_name='imu',
            preprocessed_data=preprocessed_data
        )
        fm_dict = self.stages[2](
            map_name='fm_sensor',
            preprocessed_data=preprocessed_data,
            imu_map=imu_map
        )
        sensation_map = None
        if not self.inference:
            sensation_map = self.stages[2](
                map_name='sensation',
                preprocessed_data=preprocessed_data,
                imu_map=imu_map
            )

        # Step-3: Sensor fusion
        scheme_dict = self.stages[3](fm_dict=fm_dict)

        # Step-4: Extract detections (event and non-event) from segmented data
        extracted_detections = self.stages[4](
            inference=self.inference,
            preprocessed_data=preprocessed_data,
            scheme_dict=scheme_dict
        )
        # Step-5: Extract features of each detection
        extracted_features = self.stages[5](
            inference=self.inference,
            fm_dict=fm_dict,
            extracted_detections=extracted_detections
        )

        return {
            'imu_map': imu_map,
            'fm_dict': fm_dict,
            'scheme_dict': scheme_dict,
            'sensation_map': sensation_map,
            'extracted_detections': extracted_detections,
            'extracted_features': extracted_features
        }
