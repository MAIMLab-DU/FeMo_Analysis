import numpy as np
from collections import defaultdict
from skimage.measure import label
from functools import reduce
from .base import BaseTransform


class SensorFusion(BaseTransform):

    def __init__(self,
                 desired_scheme: int = 1,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        assert desired_scheme in range(9), f"{desired_scheme} must be in range(0, 9)"
        self.desired_scheme = self.scheme_map[desired_scheme]

    def transform(self, fm_dict: dict):

        labeled_fm_map = label(fm_dict['fm_map'])
        num_labels = len(np.unique(labeled_fm_map)) - 1  # -1 to exclude background
        labeled_fm_map = labeled_fm_map.reshape((labeled_fm_map.size, ))
        print(f"Before: {num_labels = }")

        fm_segmented = fm_dict['fm_segmented']
        if not self.use_all_sensors:
            labeled_user_scheme = labeled_fm_map
        else:
            user_scheme = np.zeros_like(fm_segmented[0])
            
            if num_labels > 0:

                if self.desired_scheme[0] == 'type':
                    fm_segmented_by_type = defaultdict()
                    for key, value in self.sensor_map.items():
                        indices = [int(value[i].split('_')[1])-1 for i in range(len(value))]
                        fm_segmented_by_type[key] = [reduce(lambda x, y: x | y, (fm_segmented[i] for i in indices))]

                for idx in range(1, num_labels+1):
                    label_start = np.where(labeled_fm_map == idx)[0][0]  # start of the label
                    label_end = np.where(labeled_fm_map == idx)[0][-1] + 1  # end of the label

                    individual_map = np.zeros_like(fm_segmented[0])
                    individual_map[label_start:label_end] = 1
                    
                    # Check if detection is in at least n sensors
                    if self.desired_scheme[0] == 'number':
                        tmp_var = sum([np.any(individual_map * each_fm_segmented) \
                                       for each_fm_segmented in fm_segmented])                        
                    
                    # Check if detection is in at least n type of sensors
                    elif self.desired_scheme[0] == 'type':
                        tmp_var = sum([np.any(individual_map * each_fm_segmented_type) 
                                       for each_fm_segmented_type in fm_segmented_by_type.values()])                

                    if tmp_var >= self.desired_scheme[1]:
                        user_scheme[label_start:label_end] = 1
            
            labeled_user_scheme = label(np.array(user_scheme))
            
        num_labels = len(np.unique(labeled_user_scheme)) - 1  # -1 to exclude background
        labeled_user_scheme = labeled_user_scheme.reshape((labeled_user_scheme.size, ))
        print(f"After: {num_labels = }")
        
        return {
            'labeled_user_scheme': labeled_user_scheme,
            'num_labels': num_labels,
        }
            









