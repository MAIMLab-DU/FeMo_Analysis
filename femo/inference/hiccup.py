import numpy as np
import matplotlib.pyplot as plt
from ..data.transforms._utils import (
    custom_binary_dilation
)
from scipy.signal import find_peaks
from skimage.measure import label
import os


class HiccupAnalysis:
    '''
    The Hiccup Analysis class requires two parameters during initialization:
        "dilation_length" and "Fs_sensor"(sampling frequency)."

    Hiccup Conditions
    -----------------
    1. Recurring sharp signal occurring every 2-4s
    2. Continuous persistent hiccup movement between 1.42 - 21.20 minutes
    3. Consistent amplitude of raw signal(Piezo). (under a minimum standard deviation)

    -----------------
    "dilation_length" parameter is utilized to specify the interval of each hiccup occurance
        - default value is set to 4, indicating that all consecutive movements within a 4-second window (2 seconds in both directions) are regarded as hiccups.

    "threshold_second" in isHiccup() is used to define coninuous hiccup movement duration in second
        - threshold_second = 85 indicates all the initial hiccup detection less than 85 second will be discarded.

    "get_filtered_signal_stats()" is used to filter out raw signal(piezo) and calculate signal statistics discard inconsistant signal as hiccup

    '''

    def __init__(self, hiccup_period_distance, Fs_sensor, y_shift, fusion,
                 peak_distance, fm_dilation, hiccup_continuous_time, exception_per_minute,
                 tolerance_limit, std_threshold_percentage, delta):
        self.Fs_sensor = Fs_sensor
        self.period_distance = hiccup_period_distance
        self.y_shift = y_shift
        self.delta = delta
        self.fusion = fusion
        self.peak_distance = peak_distance
        self.fm_dilation = fm_dilation
        self.hiccup_continuous_time = hiccup_continuous_time
        self.exception_threshold = 60 / exception_per_minute
        self.tolerance_limit = tolerance_limit
        self.std_threshold_percentage = std_threshold_percentage

        self.hiccup_data_file_indices_condition1_2 = []
        self.hiccup_data_file_indices_condition1_3 = []
        self.hiccup_data_file_indices = []
        self.hiccup_data_file_indices_each = {}
        self.hiccup_map_list = []
        self.hiccup_map_list_condition1_2_4 = []
        self.std_fltd_map = []
        self.max_width_list = []
        self.condition1_data = []
        self.condition3_data = []
        self.hiccup_gt = []

        self.mean_value_list = {}
        self.std_deviation_list = {}
        self.variance_list = {}
        self.fltd_map_array = {}
        self.std_deviation_values_list = {}
        self.max_std_deviation = {}
        self.piezo_1_or_2 = {}
        self.acoustic_1_or_2 = {}
        self.accelerometer_1_or_2 = {}

    def remove_spikes_test(self, binary_signal, threshold):
        """
        Remove spikes from a binary signal based on width threshold.
        Revove spikes <= threshold width in second

        Parameters:
        - binary_signal (list or array): Binary signal (0 or 1).
        - threshold (int): Minimum width in second of spikes to be retained.

        Returns:
        - cleaned_signal (list): Binary signal after spike removal.
        - max_width (float): Max spike width of signal in second.
        """
        # If signal has no spike, retun input signal and max_width as zero
        if np.max(binary_signal) < 1:
            return (binary_signal, 0)

        threshold = self.Fs_sensor * threshold  # Convert in sample space

        labelled = label(binary_signal)  # get number of spikes labelling them in ascending order
        interested_index = []  # to hold indices where to perform dilation

        for i in range(1, max(labelled) + 1):  # from 1 to total no_of_spikes
            indexs = np.where(labelled == i)[0]  # indexs where i is found
            if len(indexs) > 1:
                # store only firts and last index
                interested_index.append(indexs[0])
                interested_index.append(indexs[-1])
            else:
                interested_index.append(indexs[0])

        zero_map = np.zeros(len(binary_signal), dtype=int)

        start_idx = interested_index[0]
        end_idx = interested_index[1]
        zero_map[start_idx:end_idx] = 1

        # for i in range(len(interested_index)):
        #     if i < len(interested_index) / 2:
        #         start_idx_position = 2 * i
        #         end_idx_position = start_idx_position + 1
        #         start_idx = interested_index[start_idx_position]
        #         end_idx = interested_index[end_idx_position]
        #         zero_map[start_idx:end_idx] = 1

        return zero_map

    def remove_spikes_retun_max(self, binary_signal, threshold):
        """
        Remove spikes from a binary signal based on width threshold.
        Revove spikes <= threshold width in second

        Parameters:
        - binary_signal (list or array): Binary signal (0 or 1).
        - threshold (int): Minimum width in second of spikes to be retained.

        Returns:
        - cleaned_signal (list): Binary signal after spike removal.
        - max_width (float): Max spike width of signal in second.
        """
        # If signal has no spike, retun input signal and max_width as zero
        if np.max(binary_signal) < 1:
            return (binary_signal, 0)

        threshold = self.Fs_sensor * threshold  # Convert in sample space

        # Find differences between consecutive elements
        diff_signal = np.diff(binary_signal, prepend=0, append=0)

        # Find indices where spikes start and end
        spike_starts = np.where(diff_signal == 1)[0]
        spike_ends = np.where(diff_signal == -1)[0]

        # Calculate spike widths
        spike_widths = spike_ends - spike_starts

        max_width = np.max(spike_widths) / self.Fs_sensor  # max width of signal in second

        # Create cleaned signal
        cleaned_signal = np.zeros(len(binary_signal), dtype=int)

        # Remove spikes below the threshold
        for start, width in zip(spike_starts, spike_widths):
            if width > threshold:
                cleaned_signal[start:start + width] = 1

        return cleaned_signal, max_width

    def remove_spikes_with_exception(self, binary_signal, threshold, exception_threshold_time, tolerance_range):
        """
        Remove spikes from a binary signal based on width threshold.
        Allow exceptions for merging spikes within a tolerance range, limited by time-based exception threshold.

        Parameters:
        - binary_signal (list or array): Binary signal (0 or 1).
        - threshold (int): Minimum width in seconds of spikes to be retained.
        - exception_threshold_time (int): Time threshold in seconds before considering new exceptions.
        - tolerance_range (int): Tolerance range in seconds to consider gaps as continuous signal.

        Returns:
        - cleaned_signal (list): Binary signal after spike removal.
        - max_width (float): Max spike width of signal in seconds.
        """
        # If signal has no spike, return input signal and max_width as zero
        if np.max(binary_signal) < 1:
            return binary_signal, 0

        threshold_samples = self.Fs_sensor * threshold  # Convert threshold to sample space
        tolerance_samples = self.Fs_sensor * tolerance_range  # Convert tolerance range to sample space
        exception_threshold_samples = self.Fs_sensor * exception_threshold_time  # Convert exception threshold to sample space

        # Find differences between consecutive elements
        diff_signal = np.diff(binary_signal, prepend=0, append=0)

        # Find indices where spikes start and end
        spike_starts = np.where(diff_signal == 1)[0]
        spike_ends = np.where(diff_signal == -1)[0]

        # Calculate spike widths
        spike_widths = spike_ends - spike_starts

        max_width = np.max(spike_widths) / self.Fs_sensor  # max width of signal in seconds

        # Create cleaned signal
        cleaned_signal = np.zeros(len(binary_signal), dtype=int)

        # Initialize variables for exception handling
        last_exception_time = -exception_threshold_samples
        i = 0

        while i < len(spike_starts):
            start = spike_starts[i]
            width = spike_widths[i]

            if width > threshold_samples:
                # If the spike width is greater than the threshold, retain it
                cleaned_signal[start:start + width] = 1
            else:
                # If the spike width is less than the threshold
                current_time = spike_ends[i]
                if current_time - last_exception_time > exception_threshold_samples:
                    # Check if the next spike is within the tolerance range
                    if i + 1 < len(spike_starts) and spike_starts[i + 1] - spike_ends[i] <= tolerance_samples:
                        # Merge the current spike with the next one
                        next_start = spike_starts[i + 1]
                        next_width = spike_widths[i + 1]
                        while i + 1 < len(spike_starts) and spike_starts[i + 1] - spike_ends[i] <= tolerance_samples:
                            i += 1
                            next_start = spike_starts[i]
                            next_width = spike_widths[i]
                        merged_width = next_start + next_width - start
                        if merged_width > threshold_samples:
                            cleaned_signal[start:next_start + next_width] = 1
                            last_exception_time = next_start + next_width
                        else:
                            # If the merged width is not greater than the threshold, discard the spike
                            last_exception_time = current_time
                    else:
                        # If the gap is larger than the tolerance range, discard the spike
                        last_exception_time = current_time
                else:
                    # If within the exception time threshold, discard the spike
                    last_exception_time = current_time

            i += 1

        return cleaned_signal, max_width
    
    def consider_body_movement_as_hiccup_v2(self, hiccup_map, IMU_map, delta):
        """
        Consider body movement as hiccup if there are two consecutive hiccups within a certain range and
        any body movement within delta time window before, after, or between the hiccups.
        Merge the hiccups if there is an IMU signal between them.

        Parameters
        ----------
        hiccup_map : Binary map of possible hiccup positions
        IMU_map : Binary map of body movement positions
        delta : Time window in seconds to check for body movement

        Returns
        -------
        cleaned_hiccup_map : Binary map of confirmed hiccup positions
        
        Notes
        -----
        Delta window is set to a specified value before and after body movement.
        Uses sampling rate self.Fs_sensor to convert time to samples.
        """
        # # If there are less than two hiccups, return the input map
        # if np.sum(hiccup_map) < 2:
        #     return hiccup_map

        # Convert delta to samples
        delta_samples = int(delta * self.Fs_sensor)

        # Find indices of hiccup occurrences
        # hiccup_indices = np.where(hiccup_map == 1)[0]

        ############ new body movement start ###########
        # Label the bouts
        labeled_map = label(hiccup_map)
        n_bouts = labeled_map.max()

        if np.sum(n_bouts) < 2:
            return hiccup_map
        
        # Create output map
        hiccup_map_body = hiccup_map.copy()
        
        # Process each bout
        for i in range(1, n_bouts):
            # Find current bout end and next bout start
            current_bout_end = np.where(labeled_map == i)[0][-1]
            next_bout_start = np.where(labeled_map == i + 1)[0][0]
            
            # If bouts are close enough, merge them
            if (next_bout_start - current_bout_end) <= delta_samples:
                # hiccup_map_body[current_bout_end:next_bout_start] = 1

                # Check if there is any body movement within the window or between the hiccups
                if np.max(IMU_map[current_bout_end:next_bout_start]) > 0:
                    # Merge the hiccups and mark the range as valid in the cleaned hiccup map
                    hiccup_map_body[current_bout_end:next_bout_start] = 1
                    # i += 1  # Skip the next hiccup since it is merged

        ######### new body movement end #############
    

        return hiccup_map_body

    
    def smooth_signal(self, y, window_size):
        box = np.ones(window_size) / window_size
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def replace_outliers_with_zero(self, data, m=5.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / mdev if mdev else np.zeros(len(d))
        data[s >= m] = 0
        return data

    def integrateData(self, signal):

        signal_shape = signal.shape[0]
        new_shape_y = 20  # 50ms time window
        new_shape_x = signal_shape // new_shape_y

        # fill up extra values with zero
        signal = np.pad(signal, (0, new_shape_y * (new_shape_x + 1) - signal_shape), mode='constant', constant_values=0)

        integrated_signal = np.sum(signal.reshape((new_shape_x + 1, new_shape_y)), axis=1)

        return integrated_signal

    def seperate_regions_of_hiccup_map(self, hiccup_map):

        """
        Parameters
        ----------
        hiccup_map : Binary map of possible hiccup positions
        Returns
        -------

        """
        labelled = label(hiccup_map)  # get number of spikes labelling them in ascending order
        interested_index = []  # to hold indices where to perform dilation
        map_segmented = []
        for i in range(1, max(labelled) + 1):  # from 1 to total no_of_spikes
            indexs = np.where(labelled == i)[0]  # indexs where i is found
            if len(indexs) > 1:
                # store only first and last index
                interested_index.append(indexs[0])
                interested_index.append(indexs[-1])
            else:
                interested_index.append(indexs[0])

        for i in range(len(interested_index)):
            if i < len(interested_index) / 2:
                start_idx_position = 2 * i
                end_idx_position = start_idx_position + 1
                start_idx = interested_index[start_idx_position]
                end_idx = interested_index[end_idx_position]
                zero_map = np.zeros(len(hiccup_map), dtype=int)
                zero_map[start_idx:end_idx] = 1
                map_segmented.append(zero_map)

        return map_segmented

    def standard_deviation_check_per_sensor_regions(self, signal, map_segmented, std_threshold):
        """
        Parameters
        ----------
        signal : 1D numpy array
            sensor filtered signal.

        Returns
        -------

        """

        # ===Build a colvoluted smoothed curve to use it as a nonlinerar threshold to detect peaks
        smoothCurve = self.smooth_signal(signal, window_size=int(signal.shape[0] * .003))
        outlier_removed = self.replace_outliers_with_zero(signal.copy(),
                                                          5)  # remove outlier to get filtered signal aplitude
        threshold = smoothCurve + abs(
            np.max(outlier_removed) - np.min(outlier_removed)) * 0.4  # y-shift 40% of signal amplitude
        std_deviation_list = []
        for i in range(len(map_segmented)):
            # Consider only positive hiccup region to calculate stats
            ROI = signal * map_segmented[i]

            # detct peaks
            peaks, _ = find_peaks(ROI, height=threshold, distance=self.peak_distance) # 800

            # Calculate mean, standard deviation, and variance
            # mean_value = np.mean(ROI[peaks])
            # variance = np.var(ROI[peaks])
            std_deviation = np.std(ROI[peaks])
            std_deviation_list.append(std_deviation)

            if std_deviation > std_threshold:
                map_segmented[i] = np.zeros_like(map_segmented[i])
        max_std_deviation = max(std_deviation_list)
        std_checked = map_segmented[0].copy()

        for array in map_segmented[1:]:
            std_checked = np.logical_or(std_checked, array)

        return std_checked, std_deviation_list, max_std_deviation
    
    def standard_deviation_check_per_sensor_regions_new(self, signal, map_segmented, std_threshold_percentage):
        """
        Parameters
        ----------
        signal : 1D numpy array
            sensor filtered signal.

        Returns
        -------

        """

        # ===Build a colvoluted smoothed curve to use it as a nonlinerar threshold to detect peaks
        # smoothCurve = self.smooth_signal(signal, window_size=int(signal.shape[0] * .003)) #previous smooting window
        smoothCurve = self.smooth_signal(signal, window_size=min(int(signal.shape[0] * 0.003), 7168)) #7168 # Cap at 7s cause large dataset close to 60 minutes takes too long to process
        
        outlier_removed = self.replace_outliers_with_zero(signal.copy(),
                                                          5)  # remove outlier to get filtered signal aplitude
        
        threshold = smoothCurve + abs(
            np.max(outlier_removed) - np.min(outlier_removed)) * self.y_shift  # y-shift 40% of signal amplitude
        std_deviation_list = []
        for i in range(len(map_segmented)):
            # Consider only positive hiccup region to calculate stats
            ROI = signal * map_segmented[i]

            # detct peaks
            peaks, _ = find_peaks(ROI, height=threshold, distance=800)

            if peaks.size == 0:  # Handle empty peaks
                std_deviation_list.append(0)
                continue

            # Calculate mean, standard deviation, and variance
            # mean_value = np.mean(ROI[peaks])
            # variance = np.var(ROI[peaks])
            # print("PEAKS= ", ROI[peaks])
            peak_range = np.max(ROI[peaks]) - np.min(ROI[peaks])
            # print("peak range= ", peak_range)
            std_threshold = std_threshold_percentage * peak_range
            # print("std threshold", std_threshold)
            std_deviation = np.std(ROI[peaks])
            
            # print("std deviation", std_deviation)
            std_deviation_list.append(std_deviation)

            if std_deviation > std_threshold:
                map_segmented[i] = np.zeros_like(map_segmented[i])

        max_std_deviation = max(std_deviation_list)
        std_checked = map_segmented[0].copy()

        for array in map_segmented[1:]:
            std_checked = np.logical_or(std_checked, array)

        return std_checked, std_deviation_list, max_std_deviation
    
    def standard_deviation_check_all_sensors(self, all_sensors, threshold_percentage, hiccup_map):

        all_sensor_map = {}
        std_deviation_list_map = {}
        max_std_deviation_map = {}

        # for sensor in all_sensors.keys():
        #         signal = all_sensors[sensor]
        #         # baseline_removed_signal = signal - np.mean(signal)  # Remove baseline
        #         # all_sensors[sensor] = np.abs(baseline_removed_signal)  # Take absolute value
            # all_sensors[sensor] = np.abs(all_sensors[sensor])

        for sensor in all_sensors.keys():
            map_segmented = self.seperate_regions_of_hiccup_map(hiccup_map)
            
            # print("sensor key = ", sensor)
            signal = all_sensors[sensor]

            all_sensor_map[sensor], std_deviation_list_map[sensor], max_std_deviation_map[
                sensor] = self.standard_deviation_check_per_sensor_regions_new(signal, map_segmented,
                                                                               threshold_percentage)

        return all_sensor_map, std_deviation_list_map, max_std_deviation_map

    def get_filtered_signal_stats(self, signal, ROI):
        """
        Parameters
        ----------
        signal : 1D numpy array
            sensor filtered signal.
        ROI : 1D numpy array
            Region of interest where to check stats

        Returns
        -------
        (mean, SD, variance): tuple
        """
        if np.max(ROI) < 1:
            mean_value = 0
            std_deviation = 0
            variance = 0
            return mean_value, std_deviation, variance

            # Remove outliers
        # signal = self.replace_outliers_with_zero(signal, 30)

        # ===Build a colvoluted smoothed curve to use it as a nonlinerar threshold to detect peaks
        smoothCurve = self.smooth_signal(signal, window_size= np.min(int(signal.shape[0] * .003), 7168))  # 7168
        outlier_removed = self.replace_outliers_with_zero(signal.copy(),
                                                          5)  # remove outlier to get filtered signal aplitude
        smoothCurve += abs(np.max(outlier_removed) - np.min(outlier_removed)) * 0.6  # y-shift 40% of signal amplitude

        # Consider only positive hiccup region to calculate stats
        signal = signal * ROI

        # detct peaks
        peaks, _ = find_peaks(signal, height=smoothCurve, distance=800) #800

        # Calculate mean, standard deviation, and variance
        mean_value = np.mean(signal[peaks])
        std_deviation = np.std(signal[peaks])
        variance = np.var(signal[peaks])

        return mean_value, std_deviation, variance
 
    def plot_data_v2(self, data_x, x_axis_type, subplotNo, title, thickness, color='blue', show_xticks=False, xticks=None, binary_signal=None):
        plt.subplot(6, 1, subplotNo)
        time_each_data_point = np.arange(0, len(data_x), 1)

        if x_axis_type == 'm':
            time_ticks = time_each_data_point / 1024 / 60
        elif x_axis_type == 's':
            time_ticks = time_each_data_point / 1024

        plt.plot(time_ticks, data_x, linewidth=thickness, color=color)

        # Add binary signal as a line or scatter plot if provided
        if binary_signal is not None:
            plt.plot(time_ticks, binary_signal * max(data_x) * 0.8, label='Individual Hiccup Map', color='green', linestyle='--', alpha=0.7)

        plt.box(False)

        if show_xticks:
            plt.xticks(xticks, fontsize=14)
        else:
            plt.xticks(xticks, labels=[])

        plt.title(title)
        plt.yticks([])

    def plot_hiccup_analysis_v3(self, fltdSignal_piezo1, fltdSignal_aclm1, fltdSignal_acstc1, 
                           detection_map, hiccup_map, x_axis_type, file_name,
                           piezo_1_or_2, acoustic_1_or_2, accelerometer_1_or_2, cndtn1_body_map):
        """Plot hiccup analysis with sensation data and ground truth"""

        total_time_points = max(len(fltdSignal_piezo1), len(fltdSignal_aclm1), 
                            len(fltdSignal_acstc1), len(detection_map), 
                            len(hiccup_map))
        time_each_data_point = np.arange(0, total_time_points, 1)

        if x_axis_type == 'm':
            time_ticks = time_each_data_point / 1024 / 60
        elif x_axis_type == 's':
            time_ticks = time_each_data_point / 1024

        # Adjust xticks for readability when time range is too large
        max_time = max(time_ticks)
        if max_time > 30:
            tick_interval = max_time // 10
        else:
            tick_interval = 3

        xticks = np.arange(0, max_time + tick_interval, tick_interval)

        # Create figure with specific size
        plt.figure(figsize=(12, 12))

        # Plot sensor data with associated binary signals and condition maps
        self.plot_data_v2(fltdSignal_piezo1, x_axis_type, 1, "Piezo Sensor Data", 2, (.2, .5, .8), xticks=xticks, binary_signal=piezo_1_or_2)
        plt.plot(time_ticks, cndtn1_body_map * max(fltdSignal_piezo1) * 0.6, label='Maternal Body Movement', color='orange', linestyle='-.', alpha=0.7)
        plt.legend(fontsize=10)

        self.plot_data_v2(fltdSignal_aclm1, x_axis_type, 2, "Accelerometer Sensor Data", 2, (.2, .5, .8), xticks=xticks, binary_signal=acoustic_1_or_2)
        self.plot_data_v2(fltdSignal_acstc1, x_axis_type, 3, "Acoustic Sensor Data", 2, (.2, .5, .8), xticks=xticks, binary_signal=accelerometer_1_or_2)

        # Plot detection map in position 4
        plt.subplot(6, 1, 4)
        plt.plot(time_ticks, detection_map, label="Detected Fetal Movements", color=(.2, 0.5, .8), linewidth=2)
        plt.ylim(-.1, 1.1)
        plt.xticks(xticks, fontsize=14)
        plt.ylabel("Fetal\nMovements", fontsize=12)
        plt.legend(fontsize=10)

        # Plot hiccup map in position 5
        plt.subplot(6, 1, 5)
        plt.plot(time_ticks, hiccup_map, label="Detected Hiccup Regions", color=(0.8, 0, 0), linewidth=2)
        plt.ylim(-.1, 1.1)
        plt.xticks(xticks, fontsize=14)
        plt.ylabel("Detected\nHiccups", fontsize=12)
        plt.legend(fontsize=10)

        plt.xlabel('Time (minutes)', fontweight='bold', fontsize=15)
        plt.ylim(-.1, 1.1)
        plt.xticks(xticks, fontsize=14)
        plt.ylabel("Ground Truth\nHiccups", fontsize=12)
        plt.legend(fontsize=10)

        plt.subplots_adjust(hspace=0.8)
        # Ensure that the output directory exists before saving
        output_dir = os.path.dirname(file_name)
        os.makedirs(output_dir, exist_ok=True)

        # Save plot
        if not file_name.endswith(".png"):
            file_name += ".png"  # Make sure file extension is included
        # Save plot
        # os.makedirs("Results/Hiccup", exist_ok=True)
        # base_name = os.path.basename(file_name)
        # output_path = os.path.join("Results", "Hiccup", f"Hiccup_{base_name}.png")
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()

    def _merge_nearby_bouts(self, binary_map, time_threshold_sec=10, sampling_rate=1024):
        """Merge bouts that are within specified time threshold of each other"""
        
        # Convert time threshold to samples
        sample_threshold = time_threshold_sec * sampling_rate
        
        # Label the bouts
        labeled_map = label(binary_map)
        n_bouts = labeled_map.max()
        if n_bouts == 0:
            return binary_map
        
        # Create output map
        merged_map = binary_map.copy()
        
        # Process each bout
        for i in range(1, n_bouts):
            # Find current bout end and next bout start
            current_bout_end = np.where(labeled_map == i)[0][-1]
            next_bout_start = np.where(labeled_map == i + 1)[0][0]
            
            # If bouts are close enough, merge them
            if next_bout_start - current_bout_end <= sample_threshold:
                merged_map[current_bout_end:next_bout_start] = 1
                
        return merged_map


    def detect_hiccups(self, reduced_detection_map, signals, imu_map):
        """
        Parameters
        ----------
        index : int
            index of data file to save hiccup index.
        reduced_detection_map : numpy array
            binary ML detection map.
        threshold_second : int
            Used to trim continuous hiccup movement less that threshold width(in second).

        Returns
        -------
        result : bool
            True if hiccup present; False otherwise
        hiccup_map : numpy array
            binary hiccup data map of provided ML detection map
        maxWidth : int
            max width of hiccup_map in second

        """

        # =======Check Condition_1==========
        dilated_data_raw_ = custom_binary_dilation(reduced_detection_map, self.period_distance * self.Fs_sensor)
        # =======Check Condition_4==========
        # considering body movement from IMU
        # if IMU_map is not None:
        # if in between two hiccup there is a body movement, consider it as a hiccup and merge them
        dilated_data_raw = self.consider_body_movement_as_hiccup_v2(dilated_data_raw_, imu_map, self.delta)
    
        # =======Check Condition_2==========
        # get binary hiccup_map map of provided ML detection map, and max_width of hiccup_map in second
        hiccup_map, max_width = self.remove_spikes_with_exception(dilated_data_raw, self.hiccup_continuous_time,
                                                                  self.exception_threshold,
                                                                  self.tolerance_limit)  # Apply hiccup frequency and continuity

        # # if hiccup is present will return hiccup is present
        # =======Check Condition_3==========
        std_all_sensors, std_deviation_values_list, max_std_deviation = self.standard_deviation_check_all_sensors(
            signals, self.std_threshold_percentage, hiccup_map
        )

        piezo_1_2_or = np.logical_or(std_all_sensors['Piezo1'], std_all_sensors['Piezo2'])
        aclm_1_2_or = np.logical_or(std_all_sensors['Aclm1'], std_all_sensors['Aclm2'])
        acstc_1_2_or = np.logical_or(std_all_sensors['Acstc1'], std_all_sensors['Acstc2'])
        
        piezo_and_aclm = np.logical_or(piezo_1_2_or, aclm_1_2_or)
        piezo_and_aclm_and_acstc = np.logical_or(piezo_and_aclm, acstc_1_2_or)

        # New Condition: At least two types of sensors should have common high
        piezo_and_aclm = np.logical_and(piezo_1_2_or, aclm_1_2_or)
        piezo_and_acstc = np.logical_and(piezo_1_2_or, acstc_1_2_or)
        aclm_and_acstc = np.logical_and(aclm_1_2_or, acstc_1_2_or)

        piezo_supremacy = np.logical_or(piezo_1_2_or, np.logical_and(aclm_1_2_or, acstc_1_2_or))
        # piezo_supremacy = piezo_1_2_or

        # Combine the conditions to ensure at least two sensor types are high
        fusion_hiccup_map_TWO = np.logical_or(np.logical_or(piezo_and_aclm, piezo_and_acstc), aclm_and_acstc)

        # New Condition: At least three types of sensors should have common high
        fusion_hiccup_map_THREE = np.logical_and(piezo_1_2_or, np.logical_and(aclm_1_2_or, acstc_1_2_or))
        if self.fusion == 'piezo_sup':
            # For piezo supremacy, use only the piezo sensors for fusion
            fusion_hiccup_map = piezo_supremacy
        elif self.fusion == 'any_type':
            fusion_hiccup_map = piezo_and_aclm_and_acstc
            
        elif self.fusion == 'two_type':
            # Use the condition where at least two types of sensors are high
            fusion_hiccup_map = fusion_hiccup_map_TWO
        elif self.fusion == 'three_type':
            # Use the condition where at least three types of sensors are high
            fusion_hiccup_map = fusion_hiccup_map_THREE
        else:
            raise ValueError("Invalid fusion mode. Please choose 'piezo_only', 'two_sensors', or 'three_sensors'.")
        fusion_hiccup_map = self._merge_nearby_bouts(fusion_hiccup_map)
                
        return dilated_data_raw_, hiccup_map, fusion_hiccup_map, aclm_1_2_or, piezo_1_2_or, acstc_1_2_or
    
    def hiccup_bout_counter(self, signal, length_threshold):
        """
        Parameters
        ----------
        hiccup_map : Binary map of possible hiccup positions
        Returns
        -------

        """
        labelled = label(signal)  # get number of spikes labelling them in ascending order
        interested_index = []  # to hold indices where to perform dilation
        hiccup_len = []
        length_threshold = self.Fs_sensor * length_threshold
        counter = 0
        flag = 0
        temp_end_index = 0
        for i in range(1, max(labelled) + 1):  # from 1 to total no_of_spikes

            indexs = np.where(labelled == i)[0]  # indexs where i is found

            if len(indexs) > 1:
                # store only firts and last index
                interested_index.append(indexs[0])
                interested_index.append(indexs[-1])
                hiccup_length = (indexs[-1] - indexs[0])
                if flag == 0:
                    counter += 1
                    temp_end_index = indexs[-1]
                    hiccup_len.append(hiccup_length/self.Fs_sensor/60)
                if flag == 1:
                    hiccup_diff = (indexs[0] - temp_end_index)
                    temp_end_index = indexs[-1]
                    if hiccup_diff > length_threshold:
                        hiccup_len.append(hiccup_length/self.Fs_sensor/60)
                        counter += 1
                    else:
                        hiccup_len.append((hiccup_length + hiccup_diff)/self.Fs_sensor/60)
                flag = 1

            else:
                interested_index.append(indexs[0])

        return counter, hiccup_len

    def hiccup_bout_table(self, length_threshold):
        signal_list = self.hiccup_data_file_indices_condition1_3
        piezo = self.piezo_1_or_2
        acoustic = self.acoustic_1_or_2
        accelerometer = self.accelerometer_1_or_2
        cmbnd = self.fltd_map_array
        piezo_count = 0
        acoustic_count = 0
        accelerometer_count = 0
        cmbnd_count = 0
        piezo_length = 0
        acoustic_length = 0
        accelerometer_length = 0
        cmbnd_length = 0
        piezo_max_list = []
        piezo_min_list = []
        acoustic_max_list = []
        acoustic_min_list = []
        accelerometer_max_list = []
        accelerometer_min_list = []
        all_max_list = []
        all_min_list = []


        for i in signal_list:
            piezo_temp, piezo_len = self.hiccup_bout_counter(piezo[i], length_threshold)
            acoustic_temp, acoustic_len = self.hiccup_bout_counter(acoustic[i], length_threshold)
            accelerometer_temp, accelerometer_len = self.hiccup_bout_counter(accelerometer[i], length_threshold)
            cmbnd_temp, cmbnd_len = self.hiccup_bout_counter(cmbnd[i], length_threshold)

            print("i = ", i)
            print("piezo duration: ", piezo_len)
            print("Acoustic duration: ", acoustic_len)
            print("Accelerometer duration: ", accelerometer_len)
            print("All duration: ", cmbnd_len)
            
            if len(piezo_len)>0:
                piezo_max_list.append(max(piezo_len))
                piezo_min_list.append(min(piezo_len))
            if len(acoustic_len) > 0:
                acoustic_max_list.append(max(acoustic_len))
                acoustic_min_list.append(min(acoustic_len))
            if len(accelerometer_len) >0:
                accelerometer_max_list.append(max(accelerometer_len))
                accelerometer_min_list.append(min(accelerometer_len))
            if len(cmbnd_len) > 0:
                all_max_list.append(max(cmbnd_len))
                all_min_list.append(min(cmbnd_len))

            piezo_count = piezo_count + piezo_temp
            acoustic_count = acoustic_count + acoustic_temp
            accelerometer_count = accelerometer_count + accelerometer_temp
            cmbnd_count = cmbnd_count + cmbnd_temp
            for j in piezo_len:
                piezo_length = piezo_length + j
            for j in acoustic_len:
                acoustic_length = acoustic_length + j
            for j in accelerometer_len:
                accelerometer_length = accelerometer_length + j
            for j in cmbnd_len:
                cmbnd_length = cmbnd_length + j


        max_p = max(piezo_max_list)
        min_p = min(piezo_min_list)
        max_acoustic = max(acoustic_max_list)
        min_acoustic = min(acoustic_min_list)
        max_acl = max(accelerometer_max_list)
        min_acl = min(accelerometer_min_list)
        max_all = max(all_max_list)
        min_all = min(all_min_list)

        piezo_length = piezo_length
        acoustic_length = acoustic_length
        accelerometer_length = accelerometer_length
        cmbnd_length = cmbnd_length

        piezo_percentage = piezo_length / (31 * 60)
        acoustic_percentage = acoustic_length / (31 * 60)
        accelerometer_percentage = accelerometer_length / (31 * 60)
        cmbnd_percentage = cmbnd_length / (31 * 60)

        piezo_incidence_rate = piezo_count / 31.0
        acoustic_incidence_rate = acoustic_count / 31.0
        accelerometer_incidence_rate = accelerometer_count / 31.0
        cmbnd_incidence_rate = cmbnd_count / 31.0
        piezo_average_bout_length = piezo_length / piezo_count
        acoustic_average_bout_length = acoustic_length / acoustic_count
        accelerometer_average_bout_length = accelerometer_length / accelerometer_count
        cmbnd_average_bout_length = cmbnd_length / cmbnd_count

        print("Piezo Hiccup Bout Count = ", piezo_count)
        print("Acoustic Hiccup Bout Count = ", acoustic_count)
        print("Accelerometer Hiccup Bout Count = ", accelerometer_count)
        print("Combined Hiccup Bout Count = ", cmbnd_count)
        print("Piezo Incidence Rate = ", piezo_incidence_rate)
        print("Acoustic Incidence Rate = ", acoustic_incidence_rate)
        print("Accelerometer Incidence Rate = ", accelerometer_incidence_rate)
        print("Combined Incidence Rate = ", cmbnd_incidence_rate)
        print("Piezo Hiccup Bout Average Duration = ", piezo_average_bout_length)
        print("Acoustic Hiccup Bout Average Duration = ", acoustic_average_bout_length)
        print("Accelerometer Hiccup Bout Average Duration = ", accelerometer_average_bout_length)
        print("Combined Hiccup Bout Average Duration = ", cmbnd_average_bout_length)
        print("Piezo Bout Percentage = ", piezo_percentage)
        print("Acoustic Sensor Bout Percentage = ", acoustic_percentage)
        print("Accelerometer Bout Percentage = ", accelerometer_percentage)
        print("Combined Bout Percentage = ", cmbnd_percentage)
        print("Piezo Min: ", min_p)
        print("Piezo Max: ", max_p)
        print("Acoustic Min: ", min_acoustic)
        print("Acoustic Max: ", max_acoustic)
        print("Accelerometer Min: ", min_acl)
        print("Accelerometer Max: ", max_acl)
        print("All Min: ", min_all)
        print("All Max: ", max_all)
        return piezo_count, acoustic_count, accelerometer_count, piezo_incidence_rate, acoustic_incidence_rate, accelerometer_incidence_rate, piezo_average_bout_length, acoustic_average_bout_length, accelerometer_average_bout_length, max_p, min_p

    def ensemble_hiccup_indices(self):
        # Ensenmble the indices after applying all 3 condition
        # Condition
        # (Piezo1 or Piezo2) and (Aclm1 or Aclm2) and (Acstc1 or Acstc2)
        hiccup_indices_each = self.hiccup_data_file_indices_each

        Piezo1_or_Piezo2 = set(hiccup_indices_each["Piezo1"]).union(set(hiccup_indices_each["Piezo2"]))
        Aclm1_or_Aclm2 = set(hiccup_indices_each["Aclm1"]).union(set(hiccup_indices_each["Aclm2"]))
        Acstc1_or_Acstc2 = set(hiccup_indices_each["Acstc1"]).union(set(hiccup_indices_each["Acstc2"]))

        self.hiccup_data_file_indices = list(Piezo1_or_Piezo2.union(Aclm1_or_Aclm2).union(Acstc1_or_Acstc2))
        return self.hiccup_data_file_indices