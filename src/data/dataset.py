from transforms import (
    DataLoader,
    DataPreprocessor,
    DataSegmentor,
    SensorFusion,
    DetectionExtractor,
    FeatureExtractor
)


# TODO: add functionality
class FeMoDataset(object):
    
    def __init__(self,
                 base_dir: str,
                 data_files: list,
                 data_loader: DataLoader,
                 data_preprocessor: DataPreprocessor,
                 data_segmentor: DataSegmentor,
                 sensor_fusion: SensorFusion,
                 detection_extractor: DetectionExtractor,
                 feature_extractor: FeatureExtractor) -> None:
        
        self._base_dir = base_dir
        self.data_files = data_files
        self.data_loader = data_loader
        self.data_preprocessor = data_preprocessor
        self.data_segmentor = data_segmentor
        self.sensor_fusion = sensor_fusion
        self.detection_extractor = detection_extractor
        self.feature_extractor = feature_extractor

