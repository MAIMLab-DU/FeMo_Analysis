import yaml
import time
from logger import LOGGER
from collections import defaultdict
from .transforms import (
    DataLoader,
    DataPreprocessor,
    SensorFusion,
    DataSegmentor,
    DetectionExtractor,
    FeatureExtractor
)


class Pipeline(object):

    @property
    def logger(self):
        return LOGGER

    def __init__(self,
                 cfg: dict|str,
                 inference: bool = False) -> None:

        self.inference = inference
        self.cfg = cfg if isinstance(cfg, dict) else self._get_pipeline_cfg(cfg)
        self.stages = self._get_stages(
            pipeline_cfg=self.cfg
        )

    @staticmethod
    def _get_pipeline_cfg(path=None):
        if path is None:
            raise ValueError("Pipeline configuration file path is required.")
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

    def process(self, filename: str, outputs: list[str] = [
        'imu_map', 'fm_dict', 'scheme_dict', 'sensation_map',
        'extracted_detections', 'extracted_features', 'loaded_data',
        'preprocessed_data'
    ]):
        start = time.time()
        
        # Step-0: Load data
        self.logger.debug(f"Loading data from {filename}")
        loaded_data = self.stages[0](filename=filename)

        # Step-1: Preprocess data (filter and trimming)
        self.logger.debug("Preprocessing data...")
        preprocessed_data = self.stages[1](loaded_data=loaded_data)

        # Step-2: Get imu_map, fm_dict (fm sensors), and sensation_dict (button)
        imu_map = None
        if 'imu_map' in outputs:
            self.logger.debug("Creating IMU accelerometer map...")
            imu_map = self.stages[2](
                map_name='imu',
                preprocessed_data=preprocessed_data
            )
        fm_dict = None
        if 'fm_dict' or 'scheme_dict' in outputs:
            self.logger.debug("Creating FeMo sensors map...")
            fm_dict = self.stages[2](
                map_name='fm_sensor',
                preprocessed_data=preprocessed_data,
                imu_map=imu_map
            )
        sensation_map = None
        if 'sensation_map' in outputs:
            self.logger.debug("Creating maternal sensation map...")
            sensation_map = None
            if not self.inference:
                sensation_map = self.stages[2](
                    map_name='sensation',
                    preprocessed_data=preprocessed_data,
                    imu_map=imu_map
                )

        # Step-3: Sensor fusion
        scheme_dict = None
        if 'scheme_dict' in outputs:
            self.logger.debug(f"Combining {self.stages[3].num_sensors} sensors map...")
            scheme_dict = self.stages[3](fm_dict=fm_dict)

        # Step-4: Extract detections (event and non-event) from segmented data
        extracted_detections = None
        if 'extracted_detections' in outputs:
            self.logger.debug("Extracting detections...")
            extracted_detections = self.stages[4](
                inference=self.inference,
                preprocessed_data=preprocessed_data,
                scheme_dict=scheme_dict,
                sensation_map=sensation_map
            )
        # Step-5: Extract features of each detection
        extracted_features = None
        if 'extracted_features' in outputs:
            self.logger.debug("Extracting features...")
            extracted_features = self.stages[5](
                inference=self.inference,
                fm_dict=fm_dict,
                extracted_detections=extracted_detections
            )

        self.logger.info(f"Pipeline process completed in {time.time() - start: 0.3f} seconds.")

        return {
            'imu_map': imu_map,
            'fm_dict': fm_dict,
            'scheme_dict': scheme_dict,
            'sensation_map': sensation_map,
            'extracted_detections': extracted_detections,
            'extracted_features': extracted_features,
            'preprocessed_data': preprocessed_data if 'preprocessed_data' in outputs else None,
            'loaded_data': loaded_data if 'loaded_data' in outputs else None
        }
