import os
import yaml
import joblib
import tarfile
import time
from ..logger import LOGGER
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

        imu_map = fm_dict = sensation_map = scheme_dict = extracted_detections = extracted_features = None

        # Step-2: Get imu_map, fm_dict (fm sensors), and sensation_dict (button)
        if 'imu_map' in outputs:
            self.logger.debug("Creating IMU accelerometer map...")
            imu_map = self.stages[2](
                map_name='imu',
                preprocessed_data=preprocessed_data
            )
        if 'fm_dict' or 'scheme_dict' in outputs:
            self.logger.debug("Creating FeMo sensors map...")
            fm_dict = self.stages[2](
                map_name='fm_sensor',
                preprocessed_data=preprocessed_data,
                imu_map=imu_map
            )
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
        if 'scheme_dict' in outputs:
            self.logger.debug(f"Combining {self.stages[3].num_sensors} sensors map...")
            scheme_dict = self.stages[3](fm_dict=fm_dict)

        # Step-4: Extract detections (event and non-event) from segmented data
        def extract_detections(fm_dict, scheme_dict, sensation_map):
            if fm_dict is None:
               fm_dict = self.stages[2](
                map_name='fm_sensor',
                preprocessed_data=preprocessed_data,
                imu_map=imu_map
            )
            if scheme_dict is None:
                scheme_dict = self.stages[3](fm_dict=fm_dict)
            if sensation_map is None:
                sensation_map = self.stages[2](
                    map_name='sensation',
                    preprocessed_data=preprocessed_data,
                    imu_map=imu_map
                )
            out =  self.stages[4](
                inference=self.inference,
                preprocessed_data=preprocessed_data,
                scheme_dict=scheme_dict,
                sensation_map=sensation_map
            )

            return out 

        if 'extracted_detections' in outputs:
            self.logger.debug("Extracting detections...")
            extracted_detections = extract_detections(
                fm_dict=fm_dict,
                scheme_dict=scheme_dict,
                sensation_map=sensation_map
            )
        # Step-5: Extract features of each detection
        if 'extracted_features' in outputs:
            self.logger.debug("Extracting features...")
            if extracted_detections is None:
                extracted_detections = extract_detections(
                    fm_dict=fm_dict,
                    scheme_dict=scheme_dict,
                    sensation_map=sensation_map
                )
            extracted_features = self.stages[5](
                inference=self.inference,
                fm_dict=fm_dict,
                extracted_detections=extracted_detections
            )

        self.logger.info(f"Pipeline process completed in {time.time() - start: 0.3f} seconds.")

        return {
            'loaded_data': loaded_data if 'loaded_data' in outputs else None,
            'preprocessed_data': preprocessed_data if 'preprocessed_data' in outputs else None,
            'imu_map': imu_map if 'imu_map' in outputs else None,
            'fm_dict': fm_dict if 'fm_dict' in outputs else None,
            'scheme_dict': scheme_dict if 'scheme_dict' in outputs else None,
            'sensation_map': sensation_map if 'sensation_map' in outputs else None,
            'extracted_detections': extracted_detections if 'extracted_detections' in outputs else None,
            'extracted_features': extracted_features if 'extracted_features' in outputs else None
        }

    def save(self, file_path):
        """Save the pipeline to a joblib file

        Args:
            file_path (str): Path to directory for saving the pipeline
        """
        
        joblib.dump(self, os.path.join(file_path, "pipeline.joblib"))
        tar = tarfile.open(os.path.join(file_path, "pipeline.tar.gz"), "w:gz")
        tar.add(os.path.join(file_path, "pipeline.joblib"), arcname="pipeline.joblib")
        tar.close()
        self.logger.debug(f"Pipeline saved to {file_path}")
