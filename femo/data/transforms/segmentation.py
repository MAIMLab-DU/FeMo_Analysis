import time
import numpy as np
import concurrent.futures
from typing import Literal
from functools import reduce
from ._utils import custom_binary_dilation, str2bool
from .base import BaseTransform


class DataSegmentor(BaseTransform):
    def __init__(
        self,
        percentile_threshold: dict = {
            "sensor_1": 0.0,
            "sensor_2": 0.0,
            "sensor_3": 0.0,
            "sensor_4": 0.0,
            "sensor_5": 0.0,
            "sensor_6": 0.0,
        },
        imu_modalities: list[Literal['imu_acceleration', 'imu_rotation', 'imu_sensorbased']] = [
            'imu_acceleration', 'imu_rotation', 'imu_sensorbased'
        ],
        num_common_sensors_imu: int = 2,
        imu_acceleration_threshold: float = 0.2,
        imu_rotation_threshold: int = 4,
        maternal_dilation_forward: int = 2,
        maternal_dilation_backward: int = 5,
        imu_dilation: int = 4,
        fm_dilation: int = 3,
        fm_min_sn: int | list[int] = 40,
        fm_signal_cutoff: float = 0.0001,
        remove_imu: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # validate imu_modalities
        assert all(modality in ['imu_acceleration', 'imu_rotation', 'imu_sensorbased'] for modality in imu_modalities), \
              f"Invalid imu modalities: {imu_modalities}. Should be one of ['imu_acceleration', 'imu_rotation', 'imu_sensorbased']."
        # validate imu_acceleration_threshold
        self.imu_modalities = imu_modalities
        self.percentile_threshold = percentile_threshold
        self.num_common_sensors_imu = num_common_sensors_imu
        self.imu_acceleration_threshold = imu_acceleration_threshold
        self.imu_rotation_threshold = imu_rotation_threshold
        self.imu_dilation = imu_dilation
        self.fm_dilation = fm_dilation
        self.maternal_dilation_forward = maternal_dilation_forward
        self.maternal_dilation_backward = maternal_dilation_backward
        self.fm_min_sn = [fm_min_sn for _ in range(self.num_sensors)] if isinstance(fm_min_sn, int) else fm_min_sn
        assert len(self.fm_min_sn) == self.num_sensors, f"{len(self.fm_min_sn) = } != {self.num_sensors = }"
        self.segmentation_signal_cutoff = [fm_signal_cutoff for _ in range(self.num_sensors)]
        self.remove_imu = remove_imu

    @property
    def imu_dilation_size(self):
        return round(self.imu_dilation * self.sensor_freq)

    @property
    def fm_dilation_size(self):
        return round(self.fm_dilation * self.sensor_freq)

    @property
    def extension_forward(self):
        return round(self.maternal_dilation_forward * self.sensor_freq)

    @property
    def extension_backward(self):
        return round(self.maternal_dilation_backward * self.sensor_freq)

    def transform(self, map_name: Literal["imu", "fm_sensor", "sensation"], *args, **kwargs):
        if map_name == "imu":
            return self.create_imu_map(*args, **kwargs)
        elif map_name == "fm_sensor":
            return self.create_fm_map(*args, **kwargs)
        else:  # map_name == 'sensation'
            return self.create_sensation_map(*args, **kwargs)

    def _create_imu_accleration_map(self, imu_accleration):
        imu_acceleration_map = np.abs(imu_accleration) >= self.imu_acceleration_threshold

        # ----------------------Dilation of IMU data-------------------------------
        # Dilate or expand the ROI's(points with value = 1) by dilation_size (half above and half below), as defined by SE
        imu_acceleration_map = custom_binary_dilation(imu_acceleration_map, self.imu_dilation_size)

        return imu_acceleration_map

    def _create_imu_rotation_map(self, imu_rotation) -> np.ndarray:
        """
        Vectorized rotation-based map: mark True where
        (|pitch|<1.5 and (|roll|>th or |yaw|>th))
        or (|pitch|>=1.5 and (|roll|>th or |yaw|>th)),
        then dilate.
        """
        Rabs = np.abs(imu_rotation['roll'].values)
        Pabs = np.abs(imu_rotation['pitch'].values)
        Yabs = np.abs(imu_rotation['yaw'].values)
        th   = self.imu_rotation_threshold

        # vectorized condition
        m1 = (Pabs < 1.5) & ((Rabs > th) | (Yabs > th))
        m2 = (Pabs >= 1.5) & ((Rabs > th) | (Yabs > th))
        mask = m1 | m2

        return custom_binary_dilation(mask, self.imu_dilation_size)

    def _create_sensor_based_imu_map(self, preprocessed_sensor_data: dict):
        imu_for_sensors = dict()

        for key in self.percentile_threshold.keys():
            imu_for_sensors[key] = custom_binary_dilation(
                np.abs(preprocessed_sensor_data[key]) >= self.percentile_threshold[key],
                self.imu_dilation_size,
            )

        imu_for_sensors_stacked = np.vstack(list(imu_for_sensors.values()))
        imu_for_sensors_sums = np.sum(imu_for_sensors_stacked, axis=0)

        return imu_for_sensors_sums >= self.num_common_sensors_imu

    def create_imu_map(self, preprocessed_data: dict) -> np.ndarray:
        """
        Create a boolean mask of body-movement epochs by combining three IMU modalities:
        1) acceleration threshold + dilation
        2) rotation threshold + dilation
        3) sensor-based threshold + dilation

        The mask is the logical OR of each modality's dilated map, with the first
        and last samples forced to False. Finally, contiguous True regions are
        expanded into full windows (any transition from False→True marks a window
        start, True→False marks its end).

        Args:
            preprocessed_data (dict):
                - 'imu_acceleration': 1D np.ndarray of floats
                - 'imu_rotation':     pd.DataFrame with columns ['roll','pitch','yaw']
                - keys in self.sensors if 'imu_sensorbased' enabled

        Returns:
            np.ndarray (bool): length-N mask where True indicates detected body movement.
        """
        import time
        import numpy as np

        tic = time.time()
        N = preprocessed_data['imu_acceleration'].shape[0]

        # 1) initialize a single boolean mask, all False
        mask = np.zeros(N, dtype=bool)

        # 2) Acceleration modality
        if 'imu_acceleration' in self.imu_modalities:
            self.logger.debug("Generating acceleration-based map")
            acc_map = self._create_imu_accleration_map(preprocessed_data['imu_acceleration'])
            mask |= acc_map
            del acc_map  # free immediately

        # 3) Rotation modality
        if 'imu_rotation' in self.imu_modalities:
            self.logger.debug("Generating rotation-based map")
            rot_map = self._create_imu_rotation_map(preprocessed_data['imu_rotation'])
            mask |= rot_map
            del rot_map

        # 4) Sensor-based modality
        if 'imu_sensorbased' in self.imu_modalities:
            self.logger.debug("Generating sensor-based IMU map")
            sb_inputs = {k: preprocessed_data[k] for k in self.percentile_threshold.keys()}
            sb_map = self._create_sensor_based_imu_map(sb_inputs)
            mask |= sb_map
            del sb_map

        # 5) ensure first and last sample are never marked
        mask[0]  = False
        mask[-1] = False

        # 6) identify transitions to build contiguous windows
        changes = np.flatnonzero(mask[:-1] != mask[1:]) + 1
        final = np.zeros_like(mask)

        # pair up starts and ends; any odd tail extends to the end
        for start, end in zip(changes[0::2], changes[1::2]):
            final[start:end] = True
        if len(changes) % 2 == 1:
            final[changes[-1]:] = True

        elapsed_ms = (time.time() - tic) * 1000
        self.logger.info(f"IMU map created in {elapsed_ms:.2f} ms")
        return final

    def _get_segmented_sensor_data(self, preprocessed_sensor_data, imu_map):
        low_signal_quantile = 0.25
        SE = np.ones((self.fm_dilation_size + 1))  # linear element necessary for dilation operation  # noqa: F841

        h = np.zeros(self.num_sensors)  # Variable for threshold
        segmented_sensor_data = [None] * self.num_sensors

        def getBinaryDialation(index):
            # Determining the threshold
            s = np.abs(preprocessed_sensor_data[index])
            LQ = np.quantile(
                s, low_signal_quantile
            )  # Returns the quantile value for low 25% (= low_signal_quantile) of the signal
            e = s[s <= LQ]  # Signal noise
            h[index] = self.fm_min_sn[index] * np.median(
                e
            )  # Threshold value. Each row will contain the threshold value for each data file

            if np.isnan(h[index]):  # Check if h = NaN. This happens when e=[], as median(e)= NaN for that case!!!
                h[index] = np.inf
            if h[index] < self.segmentation_signal_cutoff[index]:
                h[index] = np.inf  # Precaution against too noisy signal

            # Thresholding
            each_sensor_data_sgmntd = (s >= h[index]).astype(
                int
            )  # Considering the signals that are above the threshold value; data point for noise signal becomes zero and signal becomes 1

            # Exclusion of body movement data
            each_sensor_data_sgmntd *= 1 - imu_map  # Exclusion of body movement data

            # return  (binary_dilation(each_sensor_data_sgmntd, structure=SE), index)
            return (
                custom_binary_dilation(each_sensor_data_sgmntd, self.fm_dilation_size + 1),
                index,
            )

        # Create a ThreadPoolExecutor with a specified number of threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_sensors) as executor:
            # Submit the tasks to the executor and store the Future objects
            futures = [executor.submit(getBinaryDialation, j) for j in range(self.num_sensors)]
            # Retrieve the results as they become available
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        for r in results:
            segmented_sensor_data[r[1]] = r[0]

        return segmented_sensor_data, h

    def create_fm_map(self, preprocessed_data: dict, imu_map=None):
        if imu_map is None:
            imu_map = self.create_imu_map(preprocessed_data)

        self.logger.debug("Creating Sensor Segmentation map")

        tic = time.time()
        preprocessed_sensor_data = [preprocessed_data[key] for key in self.sensors]

        segmented_sensor_data, threshold = self._get_segmented_sensor_data(preprocessed_sensor_data, imu_map)

        for i in range(self.num_sensors):
            map_added = imu_map.astype(int) + segmented_sensor_data[i].astype(int)

            map_added[0] = 0
            map_added[-1] = 0

            # Find where changes from non-zero to zero or zero to non-zero occur
            changes = np.where((map_added[:-1] == 0) != (map_added[1:] == 0))[0] + 1

            # Create tuples of every two values
            windows = []
            for j in range(0, len(changes), 2):
                if j < len(changes) - 1:
                    windows.append((changes[j], changes[j + 1]))
                else:
                    windows.append((changes[j],))

            map_final = np.copy(segmented_sensor_data[i])

            for window in windows:
                start = window[0]

                if len(window) == 2:
                    end = window[1]
                    if np.any(map_added[start:end] == 2):
                        map_final[start:end] = 0
                else:
                    map_final[start:] = 1

            map_final[0] = 0
            map_final[-1] = 0

            segmented_sensor_data[i] = map_final.astype(int)

        combined_fm_map = reduce(lambda x, y: x | y, (data for data in segmented_sensor_data))
        self.logger.info(f"Sensor segmentation map created in {(time.time()-tic)*1000:.2f} ms")

        return {
            "fm_map": combined_fm_map,  # sensor_data_sgmntd_cmbd_all_sensors
            "fm_threshold": threshold,  # h
            "fm_segmented": segmented_sensor_data,  # sensor_data_sgmntd
        }

    def create_sensation_map(self, preprocessed_data: dict, imu_map: np.ndarray | None = None) -> np.ndarray:
        if imu_map is None and str2bool(self.remove_imu):
            imu_map = self.create_imu_map(preprocessed_data)

        tic = time.time()
        sens = preprocessed_data.get("sensation_data", np.array([], dtype=bool))
        M_idx = np.flatnonzero(sens)
        out = np.zeros(len(sens), dtype=bool)

        step = round(self.sensor_freq / self.sensation_freq)
        fwd = self.extension_forward
        bwd = self.extension_backward

        for L in M_idx:
            L1 = max(0, L * step - bwd)
            L2 = min(len(out), L * step + fwd + 1)
            out[L1:L2] = True
            if str2bool(self.remove_imu) and out[L1:L2].any() and imu_map[L1:L2].any():
                out[L1:L2] = False

        elapsed = (time.time() - tic) * 1000
        self.logger.info(f"Sensation map created in {elapsed:.2f} ms")
        return out
