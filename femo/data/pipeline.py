import os
import yaml
import joblib
import tarfile
import time
import pandas as pd
from typing import Any
from ..logger import LOGGER
from .transforms import (
    DataLoader,
    DataPreprocessor,
    SensorFusion,
    DataSegmentor,
    DetectionExtractor,
    FeatureExtractor,
)


class Pipeline(object):
    STAGE_NAME_MAP = {
        "load": 0,
        "preprocess": 1,
        "segment": 2,
        "fusion": 3,
        "extract_det": 4,
        "extract_feat": 5,
    }

    @property
    def logger(self):
        return LOGGER

    def __init__(self, cfg: dict | str, inference: bool = False) -> None:
        self.inference = inference
        self.cfg = cfg if isinstance(cfg, dict) else self._get_pipeline_cfg(cfg)
        self.stages = self._get_stages(pipeline_cfg=self.cfg)

    @staticmethod
    def _get_pipeline_cfg(path=None):
        if path is None:
            raise ValueError("Pipeline configuration file path is required.")
        with open(path, "r") as file:
            return yaml.safe_load(file)

    @staticmethod
    def _get_stages(pipeline_cfg: dict):
        # Build stages with name keys
        stages_named = {
            "load": DataLoader(**pipeline_cfg.get("load", {})),
            "preprocess": DataPreprocessor(**pipeline_cfg.get("preprocess", {})),
            "segment": DataSegmentor(**pipeline_cfg.get("segment", {})),
            "fusion": SensorFusion(**pipeline_cfg.get("fusion", {})),
            "extract_det": DetectionExtractor(**pipeline_cfg.get("extract_det", {})),
            "extract_feat": FeatureExtractor(**pipeline_cfg.get("extract_feat", {})),
        }
        # Add legacy numeric-indexed keys
        for name, idx in Pipeline.STAGE_NAME_MAP.items():
            stages_named[idx] = stages_named[name]
        return stages_named

    def get_stage(self, name_or_index: str | int):
        """Get stage by name or legacy numeric index."""
        if isinstance(name_or_index, int):
            return self.stages.get(name_or_index)
        if name_or_index in self.stages:
            return self.stages[name_or_index]
        if name_or_index in self.STAGE_NAME_MAP:
            return self.stages.get(self.STAGE_NAME_MAP[name_or_index])
        raise KeyError(f"No such stage: {name_or_index}")

    def set_stage_params(self, stage_name: str, **kwargs):
        """
        Change one or more parameters on a given stage, with logging of old/new values.
        """
        try:
            stage = self.get_stage(stage_name)
        except KeyError:
            raise KeyError(f"Unknown stage '{stage_name}'")

        for key, val in kwargs.items():
            if not hasattr(stage, key):
                raise AttributeError(f"Stage '{stage_name}' has no attribute '{key}'")
            old_val = getattr(stage, key)
            setattr(stage, key, val)
            self.logger.info(f"[{stage_name}] Updated '{key}': {old_val} -> {val}")

    def process(
        self,
        filename: str,
        feature_set: str = "crafted",
        outputs: str | list[str] = None,
    ) -> dict[str, Any]:
        # Define supported outputs and their direct stage dependencies
        SUPPORTED_OUTPUTS = {
            "loaded_data": {"load"},
            "preprocessed_data": {"preprocess"},
            "imu_map": {"segment"},
            "fm_dict": {"segment"},
            "sensation_map": {"segment"},
            "scheme_dict": {"fusion"},
            "extracted_detections": {"extract_det"},
            "extracted_features": {"extract_feat"},
        }
        # Define stage→stage prerequisites
        STAGE_DEPS = {
            "load": set(),
            "preprocess": {"load"},
            "segment": {"preprocess"},
            "fusion": {"segment"},
            "extract_det": {"fusion", "segment", "preprocess"},
            "extract_feat": {"extract_det", "segment"},
        }
        # Fixed execution order
        EXEC_ORDER = [
            "load",
            "preprocess",
            "segment",
            "fusion",
            "extract_det",
            "extract_feat",
        ]

        # Default outputs if none specified
        if outputs is None or outputs == 'all':
            outputs = list(SUPPORTED_OUTPUTS.keys())

        # Validate requested outputs
        outputs = set(outputs)
        unknown = outputs - set(SUPPORTED_OUTPUTS)
        if unknown:
            raise ValueError(f"Unknown outputs requested: {unknown}")

        # Compute all required stages by walking the dependency graph
        required_stages = set()
        
        def add_stage_and_deps(stage):
            if stage in required_stages:
                return
            required_stages.add(stage)
            for dep in STAGE_DEPS.get(stage, []):
                add_stage_and_deps(dep)

        for out in outputs:
            for stage in SUPPORTED_OUTPUTS[out]:
                add_stage_and_deps(stage)

        # Prepare storage for stage results
        results: dict[str, Any] = {}

        # Helper: run a stage only if required
        def run_stage(name: str):
            if name not in required_stages:
                return
            if name in results:
                return  # already run
            self.logger.info(f"Running stage '{name}'")
            if name == 'load':
                results['loaded_data'] = self.get_stage("load")(filename=filename)
            elif name == 'preprocess':
                results['preprocessed_data'] = self.get_stage("preprocess")(
                    loaded_data=results['loaded_data']
                )
            elif name == 'segment':
                # we might need to create three different maps
                if 'imu_map' in outputs or 'segment' in required_stages:
                    results.setdefault('imu_map', 
                        self.get_stage('segment')(
                            map_name='imu',
                            preprocessed_data=results['preprocessed_data']
                        )
                    )
                if 'fm_dict' in outputs or 'fusion' in required_stages or 'extract_det' in required_stages:
                    results.setdefault('fm_dict',
                        self.get_stage('segment')(
                            map_name='fm_sensor',
                            preprocessed_data=results['preprocessed_data'],
                            imu_map=results.get('imu_map')
                        )
                    )
                if ('sensation_map' in outputs or 'segment' in required_stages) and not self.inference:
                    results.setdefault('sensation_map',
                        self.get_stage('segment')(
                            map_name='sensation',
                            preprocessed_data=results['preprocessed_data'],
                            imu_map=results.get('imu_map')
                        )
                    )
            elif name == 'fusion':
                results['scheme_dict'] = self.get_stage('fusion')(
                    fm_dict=results['fm_dict']
                )
            elif name == 'extract_det':
                results['extracted_detections'] = self.get_stage('extract_det')(
                    inference=self.inference,
                    preprocessed_data=results['preprocessed_data'],
                    scheme_dict=results['scheme_dict'],
                    sensation_map=results.get('sensation_map')
                )
            elif name == 'extract_feat':
                results['extracted_features'] = self.get_stage('extract_feat')(
                    inference=self.inference,
                    fm_dict=results['fm_dict'],
                    extracted_detections=results['extracted_detections'],
                    feat=feature_set
                )

        # Execute in correct order
        start = time.time()
        for stage in EXEC_ORDER:
            run_stage(stage)
        self.logger.info(f"Pipeline finished in {time.time()-start:0.3f}s")

        # Return only what was requested
        return {out: results.get(out) for out in outputs}

    def extract_features_batch(self, filename: str) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
        """
        Efficiently extract multiple feature sets without re-running upstream stages.

        Returns:
            dict[str, pd.DataFrame]: Mapping of feature set names to extracted feature DataFrames
        """
        # Run pipeline once up to extract_det
        intermediate_outputs = self.process(
            filename=filename,
            outputs=[
                'loaded_data',
                'preprocessed_data',
                'imu_map',
                'fm_dict',
                'sensation_map',
                'scheme_dict',
                'extracted_detections'
            ]
        )

        extract_feat_stage = self.get_stage('extract_feat')
        results = {}

        for feature_set in extract_feat_stage.feature_sets:
            feats = extract_feat_stage(
                inference=self.inference,
                fm_dict=intermediate_outputs['fm_dict'],
                extracted_detections=intermediate_outputs['extracted_detections'],
                feat=feature_set
            )
            results[feature_set] = feats

        return results, intermediate_outputs
    
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
