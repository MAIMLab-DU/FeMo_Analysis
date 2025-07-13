import time
import numpy as np
from collections import defaultdict
from typing import Tuple, Literal
from skimage.measure import label
from functools import reduce
from .base import BaseTransform


class SensorFusion(BaseTransform):

    def __init__(self,
                 desired_scheme: Tuple[Literal['type', 'number'], int] = ['type', 1],
                 **kwargs) -> None:
        super().__init__(**kwargs)

        if not isinstance(desired_scheme, list) or len(desired_scheme) != 2:
            raise ValueError("desired_scheme must be a tuple of two elements: ('type'/'number', int)")        
        if desired_scheme[0] not in ('type', 'number'):
            raise ValueError("The first element of desired_scheme must be 'type' or 'number'")        
        if not isinstance(desired_scheme[1], int):
            raise ValueError("The second element of desired_scheme must be an integer")
        
        self.desired_scheme = tuple(desired_scheme)

    def transform(self, fm_dict: dict):

        self.logger.debug(f"Starting sensor fusion based on scheme: {self.desired_scheme}")
        tic = time.time()

        labeled_fm_map = label(fm_dict['fm_map'])
        num_labels = len(np.unique(labeled_fm_map)) - 1  # -1 to exclude background
        self.logger.debug(f"Total labeled segments in fm_map: {num_labels}")
        labeled_fm_map = labeled_fm_map.reshape((labeled_fm_map.size, ))

        fm_segmented = fm_dict['fm_segmented']
        if not self.use_all_sensors:
            self.logger.debug("Using fm_map directly as (use_all_sensors=False)")
            user_scheme = fm_dict['fm_map']
            labeled_user_scheme = labeled_fm_map
        else:
            self.logger.debug(f"Evaluating user scheme with strategy: {self.desired_scheme}")
            user_scheme = np.zeros_like(fm_segmented[0])

            if num_labels > 0:
                if self.desired_scheme[0] == 'type':
                    self.logger.debug("Organizing fm_segmented signals by sensor type")
                    fm_segmented_by_type = defaultdict()
                    for key, value in self.sensor_map.items():
                        indices = [int(value[i].split('_')[1]) - 1 for i in range(len(value))]
                        fm_segmented_by_type[key] = [reduce(lambda x, y: x | y, (fm_segmented[i] for i in indices))]
                    self.logger.debug(f"Segmented sensor types: {list(fm_segmented_by_type.keys())}")

                for idx in range(1, num_labels + 1):
                    label_start = np.where(labeled_fm_map == idx)[0][0]
                    label_end = np.where(labeled_fm_map == idx)[0][-1] + 1

                    individual_map = np.zeros_like(fm_segmented[0])
                    individual_map[label_start:label_end] = 1

                    if self.desired_scheme[0] == 'number':
                        tmp_var = sum([np.any(individual_map * each_fm_segmented) for each_fm_segmented in fm_segmented])
                    elif self.desired_scheme[0] == 'type':
                        tmp_var = sum([np.any(individual_map * each_fm_segmented_type)
                                    for each_fm_segmented_type in fm_segmented_by_type.values()])

                    if tmp_var >= self.desired_scheme[1]:
                        user_scheme[label_start:label_end] = 1

            labeled_user_scheme = label(np.array(user_scheme))

        num_labels = len(np.unique(labeled_user_scheme)) - 1
        self.logger.debug(f"Final number of labels in user scheme: {num_labels}")
        labeled_user_scheme = labeled_user_scheme.reshape((labeled_user_scheme.size, ))

        self.logger.info(f"Sensor fusion completed in {(time.time() - tic)*1000:.2f} ms")
        return {
            'user_scheme': user_scheme,
            'labeled_user_scheme': labeled_user_scheme,
            'num_labels': num_labels,
        }

            









