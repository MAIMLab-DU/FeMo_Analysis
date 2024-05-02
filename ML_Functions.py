# -*- coding: utf-8 -*-
import numpy as np
from scipy.fft import fft
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score
import tensorflow as tf
import os
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from SP_Functions import get_IMU_map,get_segmented_data


def extract_detections(M_sntn_map, sensor_data_fltd, sensor_data_sgmntd, sensor_data_sgmntd_cmbd_all_sensors_labeled, n_label, n_FM_sensors):
    # Input variables: M_sntn_map - Column vector
    #                  sensor_data_fltd - Cell variable
    #                  sensor_data_sgmntd - Cell variable
    #                  sensor_data_sgmntd_cmbd_all_sensors_labeled - Array
    #                  n_label - Scalar
    #                  n_FM_sensors - Scalar
    # Output variables: current_file_TPD_extraction_cell - Cell variable
    #                   current_file_FPD_extraction_cell - Cell variable
    #                   current_file_extracted_TPD_weightage - Array
    #                   current_file_extracted_FPD_weightage - Array

    n_candidate_TPD = len(np.unique(sensor_data_sgmntd_cmbd_all_sensors_labeled * M_sntn_map)) - 1  # Number of detections that intersect with maternal sensation
    n_candidate_FPD = n_label - n_candidate_TPD
    
    
    current_file_TPD_extraction_cell = []# Each element will contain 1 TPD for all 6 sensors
    current_file_FPD_extraction_cell = []

    current_file_extracted_TPD_weightage = []
    current_file_extracted_FPD_weightage = []

    k_TPD = 0
    k_FPD = 0
    for k in range(1, n_label+1):
        L_min = np.where(sensor_data_sgmntd_cmbd_all_sensors_labeled == k)[0][0]  # Sample no. corresponding to the start of the label
        L_max = np.where(sensor_data_sgmntd_cmbd_all_sensors_labeled == k)[0][-1]  # Sample no. corresponding to the end of the label
        indv_window = np.zeros(len(M_sntn_map))  # This window represents the current detection, which will be compared with M_sntn_map to find if it is TPD or FPD
        indv_window[L_min:L_max + 1] = 1

        # ------- Number of types of sensors the segment is detected by -----------
        
        tmp_var = 0  # Variable to hold the number of common sensors for each detection
        if n_FM_sensors == 6:  # Only when all the sensors were selected for the analysis
            # First fuse the left and right sensors of each type using OR
            sensor_data_sgmntd_Aclm = np.array(sensor_data_sgmntd[0]) | np.array(sensor_data_sgmntd[1])
            sensor_data_sgmntd_PzpltS = np.array(sensor_data_sgmntd[3]) | np.array(sensor_data_sgmntd[4])
            sensor_data_sgmntd_PzpltL = np.array(sensor_data_sgmntd[2]) | np.array(sensor_data_sgmntd[5])

            # Combine all the types of sensors in a single variable
            sensor_data_sgmntd_cmbd_multi_type_sensors_OR = [sensor_data_sgmntd_Aclm, sensor_data_sgmntd_PzpltS, sensor_data_sgmntd_PzpltL]

            # Find the types of sensors detected that as a movement
            #   tmp_var = 0; % Variable to hold number of common sensors for each detection
            for j in range(n_FM_sensors // 2):
                if np.sum(indv_window * sensor_data_sgmntd_cmbd_multi_type_sensors_OR[j]):  # Non-zero value indicates intersection
                    tmp_var += 1

        # ------ Extracting current window as TPD or FPD class ---------------%
        
        # Checking the overlap with maternal sensation
        X = np.sum(indv_window * M_sntn_map)  # Non-zero values will indicate overlap
        if X:  # There is an overlap, which means TPD class
            k_TPD += 1
            current_TPD_extraction = np.zeros((L_max - L_min + 1, n_FM_sensors))
            for j in range(n_FM_sensors):
                current_TPD_extraction[:, j] = sensor_data_fltd[j][L_min:L_max + 1]  # Each sensor in each column
            
            #print(k_TPD)
            current_file_TPD_extraction_cell.append(current_TPD_extraction)
            current_file_extracted_TPD_weightage.append(tmp_var)
            
        else:
            k_FPD += 1
            current_FPD_extraction = np.zeros((L_max - L_min + 1, n_FM_sensors))
            for j in range(n_FM_sensors):
                current_FPD_extraction[:, j] = sensor_data_fltd[j][L_min:L_max + 1]
            current_file_FPD_extraction_cell.append(current_FPD_extraction)
            current_file_extracted_FPD_weightage.append(tmp_var)
            
    return (
        current_file_TPD_extraction_cell,
        current_file_FPD_extraction_cell,
        current_file_extracted_TPD_weightage,
        current_file_extracted_FPD_weightage,
    )


def extract_detections_modified(sensor_data_fltd, sensor_data_sgmntd, sensor_data_sgmntd_cmbd_all_sensors_labeled, n_label, n_FM_sensors):
    # Input variables: M_sntn_map - Column vector
    #                  sensor_data_fltd - Cell variable
    #                  sensor_data_sgmntd - Cell variable
    #                  sensor_data_sgmntd_cmbd_all_sensors_labeled - Array
    #                  n_label - Scalar
    #                  n_FM_sensors - Scalar
    # Output variables: current_file_TPD_extraction_cell - Cell variable
    #                   current_file_FPD_extraction_cell - Cell variable
    #                   current_file_extracted_TPD_weightage - Array
    #                   current_file_extracted_FPD_weightage - Array

    n_candidate = len(np.unique(sensor_data_sgmntd_cmbd_all_sensors_labeled)) - 1  # Number of detections that intersect with maternal sensation
    # n_candidate_FPD = n_label - n_candidate_TPD
    
    
    # current_file_TPD_extraction_cell = []# Each element will contain 1 TPD for all 6 sensors
    # current_file_FPD_extraction_cell = []

    # current_file_extracted_TPD_weightage = []
    # current_file_extracted_FPD_weightage = []
    
    extracted_data_cell = []

    # k_TPD = 0
    # k_FPD = 0
    
    k_det = 0
    
    for k in range(1, n_label+1):
        L_min = np.where(sensor_data_sgmntd_cmbd_all_sensors_labeled == k)[0][0]  # Sample no. corresponding to the start of the label
        L_max = np.where(sensor_data_sgmntd_cmbd_all_sensors_labeled == k)[0][-1]  # Sample no. corresponding to the end of the label
        # indv_window = np.zeros(len(M_sntn_map))  # This window represents the current detection, which will be compared with M_sntn_map to find if it is TPD or FPD
        # indv_window[L_min:L_max + 1] = 1

        k_det = k_det + 1
        
        extracted_data = np.zeros((L_max - L_min + 1, n_FM_sensors))
        
        # Assuming sensor_data_fltd is a list of arrays
        for j in range(n_FM_sensors):
            extracted_data[:, j] = sensor_data_fltd[j][L_min:L_max + 1]
        
        # Assuming extracted_data_cell is a list
        extracted_data_cell.append(extracted_data)
        
            
    return extracted_data_cell


def get_frequency_mode(data, Fs, trim_length):
    """
    This function returns the main frequency mode of the input data.
    
    Arguments:
    data -- a vector representing the data that will be analyzed
    Fs -- a scalar value representing the sampling frequency
    trim_length -- length of trimming in Hz. This is used to get the frequency mode above a certain frequency.
    
    Returns:
    spectrum -- A vector of Fourier coefficient values
    f_vector -- Corresponding frequency vector
    f_mode -- a scalar value representing the main frequency mode
    """
    
    # -----------Generating the frequency vector of the spectrum---------------
    
    L = len(data)#data is a column vector
    freq_trim_indx = int(np.ceil(trim_length * L / Fs + 1)) # Index upto which trimming needs to be done.
#                                                           L/Fs gives number of points/Hz in the frequency spectrum.
#                                                           When multiplied by trim_length, it gives the index
#                                                           upto which trimming is necessary 
    freq_vector = Fs * np.arange(int(L / 2) + 1) / L # There are L/2 points in the single-sided spectrum.
#                                                       Each point will be Fs/L apart.
    freq_vector_trimmed = freq_vector[freq_trim_indx:] #frequency vector after trimming
    
    
    # --------------------------Fourier transform------------------------------
    FT_2_sided = np.abs(fft(data)) / L# Two-sided Fourier spectrum. Normalizing by L is generally performed during fft so that it is not neede for inverse fft
    FT_data_1_sided = FT_2_sided[:int(L / 2) + 1] #Floor is used for the cases where L is an odd number
    FT_data_1_sided[1:-1] = 2 * FT_data_1_sided[1:-1] #multiplication by 2 is used to maintain the conservation of energy
    FT_data_1_sided_trimmed = FT_data_1_sided[freq_trim_indx:]
    
    # ---------------------------Main frequency mode---------------------------
    I = np.argmax(FT_data_1_sided_trimmed)  #I is the corresponding index
    freq_mode_data = freq_vector_trimmed[I] #cell matrix with main frequency mode
    
    # ---------------------------Finalizing the output variables---------------
    f_vector = freq_vector_trimmed
    spectrum = FT_data_1_sided_trimmed
    f_mode = freq_mode_data
    
    return spectrum, f_vector, f_mode



def getPSD(data, Fs, min_freq, max_freq):
    """
    This function returns the energy of the input data within the given frequency range.
    
    Arguments:
    data -- a vector representing the data that will be analyzed
    Fs -- a scalar value representing the sampling frequency
    min_freq -- minimum value of the frequency range in Hz
    max_freq -- maximum value of the frequency range in Hz
    
    Returns:
    energy_quantifier -- ratio of the energy within the frequency range to the total energy (1-30 Hz)
    """
    
    n_sample = len(data)
    
    FFTX = fft(data)# Y = fft(X) computes the discrete Fourier transform (DFT) of X using a fast Fourier transform (FFT) algorithm.
    power = np.abs(FFTX[:int(n_sample / 2) + 1 ]) ** 2 # = (abs(FFTX(1:floor(numsample/2+1))).^2. %Power: magnitude^2
    freq_bin = np.linspace(0, Fs/2, int(n_sample/2)+1) #Computing the corresponding frequency values
    
    energy_range = np.sum(power[(min_freq <= freq_bin) & (freq_bin <= max_freq)])# Sum of the Energies within min_freq-max_freq Hz
    energy_whole = np.sum(power[(1 <= freq_bin) & (freq_bin <= 30)])# Total energy within Low Frequency band of 0-30 hz
    
    # energy_quantifier = energy_range / energy_whole # Output variable
    energy_quantifier = energy_range # Output variable
    
    return energy_quantifier


def extract_features(TPD_extracted, extracted_TPD_weightage, FPD_extracted, extracted_FPD_weightage, threshold, Fs_sensor, n_FM_sensors):
    # Total duration of TPDs and FPDs
    n_TPD = 0  # Total no. of TPD segment in all datafile selected
                # This will be less than TPD from the thresholding algorithm as each one
                # may hold multiple maternal sensation detection
    n_FPD = 0  # Total no. of FPD segment in all datafile selected
    total_TPD_data = 0  # Total TP datapoints in a data file
    total_FPD_data = 0  # Total FP datapoints in a data file

    for i in range(len(TPD_extracted)):#For each data file
        n_TPD += len(TPD_extracted[i])
        n_FPD += len(FPD_extracted[i])

        for j in range(len(TPD_extracted[i])): #For each segment count how many TP data points
            total_TPD_data += len(TPD_extracted[i][j])

        for j in range(len(FPD_extracted[i])):
            total_FPD_data += len(FPD_extracted[i][j])#For each segment count how many FP data points

    total_duration_TPD = total_TPD_data/Fs_sensor/3600  # Duration of TPD in hrs
    total_duration_FPD = total_FPD_data/Fs_sensor/3600  # Duration of FPD in hrs
    
    
    
    n_common_features = 16 #i.e. max, mean, sum, sd, percentile, skew, kurtosis, .....
    total_features = n_FM_sensors*n_common_features + 1 #Extra one feature for segment duration
    
    
    # Extraction of features from TPDs and FPDs
    X_TPD = np.zeros((n_TPD, total_features))  # Features of TPD
    X_FPD = np.zeros((n_FPD, total_features))  # Feature of FPD
    index_TPD = 0
    index_FPD = 0

    for i in range(len(TPD_extracted)):#For each data file
        for j in range(len(TPD_extracted[i])):#For each TP segment in a data file
            
            #X_TPD[index_TPD,1] = data_file_GA[i]
            #X_TPD[index_TPD,2] = extracted_TPD_weightage[i][j]#  Duration of each TPD in s
            X_TPD[index_TPD, 0] = len(TPD_extracted[i][j]) / Fs_sensor  # Duration of each TPD in s

            for k in range(n_FM_sensors):
                S = TPD_extracted[i][j][:, k]
                S_thd = np.abs(S) - threshold[i, k]
                S_thd_above = S_thd[S_thd > 0]

                # Time domain features
                X_TPD[index_TPD, k * n_common_features + 1] = np.max(S_thd)  # Max value
                X_TPD[index_TPD, k * n_common_features + 2] = np.mean(S_thd)  # Mean value
                X_TPD[index_TPD, k * n_common_features + 3] = np.sum(S_thd ** 2)  # Energy
                X_TPD[index_TPD, k * n_common_features + 4] = np.std(S_thd)  # Standard deviation
                X_TPD[index_TPD, k * n_common_features + 5] = np.percentile(S_thd, 75) - np.percentile(S_thd, 25)  # Interquartile range
                X_TPD[index_TPD, k * n_common_features + 6] = skew(S_thd)  # Skewness
                X_TPD[index_TPD, k * n_common_features + 7] = kurtosis(S_thd)  # Kurtosis

                if len(S_thd_above) == 0:
                    X_TPD[index_TPD, k * n_common_features + 8]  = 0  # Duration above threshold
                    X_TPD[index_TPD, k * n_common_features + 9]  = 0  # Mean above threshold value
                    X_TPD[index_TPD, k * n_common_features + 10] = 0  # Energy above threshold value
                else:
                    X_TPD[index_TPD, k * n_common_features + 8]  = len(S_thd_above)  # Duration above threshold
                    X_TPD[index_TPD, k * n_common_features + 9]  = np.mean(S_thd_above)  # Mean above threshold
                    X_TPD[index_TPD, k * n_common_features + 10] = np.sum(S_thd_above ** 2)  # Energy above threshold

                # Frequency domain features
                _, _, X_TPD[index_TPD, k * n_common_features + 11] = get_frequency_mode(S, Fs_sensor, 1)  # Gives the main frequency mode above 1 Hz
                X_TPD[index_TPD, k * n_common_features + 12] = getPSD(S, Fs_sensor, 1, 2)
                X_TPD[index_TPD, k * n_common_features + 13] = getPSD(S, Fs_sensor, 2, 5)
                X_TPD[index_TPD, k * n_common_features + 14] = getPSD(S, Fs_sensor, 5, 10)
                X_TPD[index_TPD, k * n_common_features + 15] = getPSD(S, Fs_sensor, 10, 20)
                X_TPD[index_TPD, k * n_common_features + 16] = getPSD(S, Fs_sensor, 20, 30)

            index_TPD += 1

    for i in range(len(FPD_extracted)):
        for j in range(len(FPD_extracted[i])):
            
            #       X_FPD[index_FPD,1] = data_file_GA[i]
            #       X_FPD[index_FPD,2] = extracted_FPD_weightage[i][j] # Duration of each TPD in s
            
            X_FPD[index_FPD, 0] = len(FPD_extracted[i][j]) / Fs_sensor  # Duration in s
            
            
            # Feature extraction for left accelerometer
            for k in range(n_FM_sensors):
                S = FPD_extracted[i][j][:, k]
                S_thd = np.abs(S) - threshold[i, k]
                S_thd_above = S_thd[S_thd > 0]

                X_FPD[index_FPD, k * n_common_features + 1] = np.max(S_thd)  # Max value
                X_FPD[index_FPD, k * n_common_features + 2] = np.mean(S_thd)  # Mean value
                X_FPD[index_FPD, k * n_common_features + 3] = np.sum(S_thd ** 2)  # Energy
                X_FPD[index_FPD, k * n_common_features + 4] = np.std(S_thd)  # Standard deviation
                X_FPD[index_FPD, k * n_common_features + 5] = np.percentile(S_thd, 75) - np.percentile(S_thd, 25)  # Interquartile range
                X_FPD[index_FPD, k * n_common_features + 6] = skew(S_thd)  # Skewness
                X_FPD[index_FPD, k * n_common_features + 7] = kurtosis(S_thd)  # Kurtosis

                if len(S_thd_above) == 0:
                    X_FPD[index_FPD, k * n_common_features + 8]  = 0  # Duration above threshold
                    X_FPD[index_FPD, k * n_common_features + 9]  = 0  # Mean above threshold value
                    X_FPD[index_FPD, k * n_common_features + 10] = 0  # Energy above threshold value
                else:
                    X_FPD[index_FPD, k * n_common_features + 8]  = len(S_thd_above)  # Duration above threshold
                    X_FPD[index_FPD, k * n_common_features + 9]  = np.mean(S_thd_above)  # Mean above threshold
                    X_FPD[index_FPD, k * n_common_features + 10] = np.sum(S_thd_above ** 2)  # Energy above threshold

                _, _, X_FPD[index_FPD, k * n_common_features + 11] = get_frequency_mode(S, Fs_sensor, 1)  # Gives the main frequency mode above 1 Hz
                X_FPD[index_FPD, k * n_common_features + 12] = getPSD(S, Fs_sensor, 1, 2)
                X_FPD[index_FPD, k * n_common_features + 13] = getPSD(S, Fs_sensor, 2, 5)
                X_FPD[index_FPD, k * n_common_features + 14] = getPSD(S, Fs_sensor, 5, 10)
                X_FPD[index_FPD, k * n_common_features + 15] = getPSD(S, Fs_sensor, 10, 20)
                X_FPD[index_FPD, k * n_common_features + 16] = getPSD(S, Fs_sensor, 20, 30)

            index_FPD += 1

    return X_TPD, X_FPD, n_TPD, n_FPD, total_duration_TPD, total_duration_FPD


def extract_features_modified(extracted_data, threshold, Fs_sensor, n_FM_sensors):
    # Total duration of TPDs and FPDs
    n_extracted = 0
    
    for i in range(len(extracted_data)):#For each data file
        n_extracted += len(extracted_data[i])
    

    
    n_common_features = 16 #i.e. max, mean, sum, sd, percentile, skew, kurtosis, .....
    total_features = n_FM_sensors*n_common_features + 1 #Extra one feature for segment duration
    
    
    # Extraction of features from TPDs and FPDs
    # X_extracted = np.zeros((n_extracted, total_features)) 
    X_extracted = np.zeros((n_extracted, total_features))
    # X_FPD = np.zeros((n_FPD, total_features))  # Feature of FPD
    index_extracted = 0

    for i in range(len(extracted_data)):#For each data file
        for j in range(len(extracted_data[i])):
            
            #X_TPD[index_TPD,1] = data_file_GA[i]
            #X_TPD[index_TPD,2] = extracted_TPD_weightage[i][j]#  Duration of each TPD in s
            #X_TPD[index_TPD, 0] = len(TPD_extracted[i][j]) / Fs_sensor  # Duration of each TPD in s
            X_extracted[index_extracted,0] = len(extracted_data[i][j]) / Fs_sensor 
            
            for k in range(n_FM_sensors):
                S = extracted_data[i][j][:, k]
                S_thd = np.abs(S) - threshold[i, k]
                S_thd_above = S_thd[S_thd > 0]

                # Time domain features
                X_extracted[index_extracted, k * n_common_features + 1] = np.max(S_thd)  # Max value
                X_extracted[index_extracted, k * n_common_features + 2] = np.mean(S_thd)  # Mean value
                X_extracted[index_extracted, k * n_common_features + 3] = np.sum(S_thd ** 2)  # Energy
                X_extracted[index_extracted, k * n_common_features + 4] = np.std(S_thd)  # Standard deviation
                X_extracted[index_extracted, k * n_common_features + 5] = np.percentile(S_thd, 75) - np.percentile(S_thd, 25)  # Interquartile range
                X_extracted[index_extracted, k * n_common_features + 6] = skew(S_thd)  # Skewness
                X_extracted[index_extracted, k * n_common_features + 7] = kurtosis(S_thd)  # Kurtosis

                if len(S_thd_above) == 0:
                    X_extracted[index_extracted, k * n_common_features + 8]  = 0  # Duration above threshold
                    X_extracted[index_extracted, k * n_common_features + 9]  = 0  # Mean above threshold value
                    X_extracted[index_extracted, k * n_common_features + 10] = 0  # Energy above threshold value
                else:
                    X_extracted[index_extracted, k * n_common_features + 8]  = len(S_thd_above)  # Duration above threshold
                    X_extracted[index_extracted, k * n_common_features + 9]  = np.mean(S_thd_above)  # Mean above threshold
                    X_extracted[index_extracted, k * n_common_features + 10] = np.sum(S_thd_above ** 2)  # Energy above threshold

                # Frequency domain features
                _, _, X_extracted[index_extracted, k * n_common_features + 11] = get_frequency_mode(S, Fs_sensor, 1)  # Gives the main frequency mode above 1 Hz
                X_extracted[index_extracted, k * n_common_features + 12] = getPSD(S, Fs_sensor, 1, 2)
                X_extracted[index_extracted, k * n_common_features + 13] = getPSD(S, Fs_sensor, 2, 5)
                X_extracted[index_extracted, k * n_common_features + 14] = getPSD(S, Fs_sensor, 5, 10)
                X_extracted[index_extracted, k * n_common_features + 15] = getPSD(S, Fs_sensor, 10, 20)
                X_extracted[index_extracted, k * n_common_features + 16] = getPSD(S, Fs_sensor, 20, 30)

            index_extracted += 1


    return X_extracted, n_extracted




def normalize_features(X):
    
    #FEATURENORMALIZE Normalizes the features in X 
    #   FEATURENORMALIZE(X) returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1. This is often a good preprocessing step to do when
    #   working with learning algorithms.


    # Calculate the mean value of each feature
    mu = np.mean(X, axis=0)

    # Subtract the mean from each feature
    X_norm = X - mu

    # Calculate the deviation (max - min) of each feature
    dev = np.max(X_norm, axis=0) - np.min(X_norm, axis=0)

    # Divide each feature by its deviation
    X_norm = X_norm / dev

    # Return the normalized features, mean, and deviation
    return X_norm, mu, dev


def divide_by_holdout(X_TPD_norm, X_FPD_norm, training_proportion):
    """
    DIVIDE_BY_HOLDOUT

    Divides the data into training and test datasets using holdout method.

    Parameters:
    - X_TPD_norm: Normalized TPD data (numpy array)
    - X_FPD_norm: Normalized FPD data (numpy array)
    - training_proportion: Proportion of data to be used for training (float)

    Returns:
    - X_train: Training data (numpy array)
    - Y_train: Training labels (numpy array)
    - X_test: Test data (numpy array)
    - Y_test: Test labels (numpy array)
    - n_training_data_TPD: Number of training TPD data samples (int)
    - n_training_data_FPD: Number of training FPD data samples (int)
    - n_test_data_TPD: Number of test TPD data samples (int)
    - n_test_data_FPD: Number of test FPD data samples (int)
    """

    # Randomization of the data sets
    n_data_TPD = X_TPD_norm.shape[0]
    np.random.seed(0)  # Uses the default random seed.
    rand_num_TPD = np.random.permutation(n_data_TPD)
    X_TPD_norm_rand = X_TPD_norm[rand_num_TPD, :]

    n_data_FPD = X_FPD_norm.shape[0]
    np.random.seed(0)  # Uses the default random seed.
    rand_num_FPD = np.random.permutation(n_data_FPD)
    X_FPD_norm_rand = X_FPD_norm[rand_num_FPD, :]

    # Dividing data into train and test datasets
    n_training_data_TPD = int(n_data_TPD * training_proportion)
    n_training_data_FPD = int(n_data_FPD * training_proportion)
    n_test_data_TPD = n_data_TPD - n_training_data_TPD
    n_test_data_FPD = n_data_FPD - n_training_data_FPD

    X_train = np.vstack((X_TPD_norm_rand[:n_training_data_TPD, :], X_FPD_norm_rand[:n_training_data_FPD, :]))
    Y_train = np.zeros((X_train.shape[0], 1))
    Y_train[:n_training_data_TPD] = 1

    X_test = np.vstack((X_TPD_norm_rand[n_training_data_TPD:, :], X_FPD_norm_rand[n_training_data_FPD:, :]))
    Y_test = np.zeros((X_test.shape[0], 1))
    Y_test[:n_test_data_TPD] = 1

    return X_train, Y_train, X_test, Y_test, n_training_data_TPD, n_training_data_FPD, n_test_data_TPD, n_test_data_FPD


def ML_prediction(data_format, X_extracted_norm, model_filepath):
    

    if data_format == '1':
        index_top_features = np.array([90,67,70,19,83,54,55,35,51,71,22,7,1,39,60,3,44,28,21,87,6,57,74,23,38,86,69,18,58,20])
        index_top_features = index_top_features-1
        X_extracted_norm_ranked = X_extracted_norm[:, index_top_features]

        import pickle
        model_filepath = "Model_folder/rf_model_selected_1.pkl"
        with open(model_filepath, 'rb') as f:
            model = pickle.load(f)

        predicted_labels = model.predict(X_extracted_norm_ranked)

        print('Prediction Done')

    elif data_format == '2':
        
        import pickle
        with open(model_filepath, 'rb') as f:
            model = pickle.load(f)

        predicted_labels = model.predict(X_extracted_norm)

        print('\n#### Prediction Result ####')

    return predicted_labels



def divide_by_K_folds(X_TPD_norm, X_FPD_norm, data_div_option, K):
    # Division into stratified K-fold
    #   2: K-fold with original ratio of FPD and TPD in each fold,
    #   3: K-fold with custom ratio of FPD and TPD in each fold.
    #   However, each of these processes will create stratified division, i.e.
    #   each fold will have the same ratio of FPD and TPD.

    # Randomization of the data sets
    n_data_TPD = X_TPD_norm.shape[0]
    np.random.seed(0)  # sets random seed to default value. This is necessary for reproducibility.
    rand_num_TPD = np.random.permutation(n_data_TPD)
    X_TPD_norm_rand = X_TPD_norm[rand_num_TPD, :]

    n_data_FPD = X_FPD_norm.shape[0]
    np.random.seed(0)  # sets random seed to default value. This is necessary for reproducibility.
    rand_num_FPD = np.random.permutation(n_data_FPD)
    X_FPD_norm_rand = X_FPD_norm[rand_num_FPD, :]

    custom_FPD_TPD_ratio = 1  # FPD/TPD ratio in case of option 3

    # Determining the data in each folds
    n_data_TPD_each_fold = n_data_TPD // K
    n_data_TPD_last_fold = n_data_TPD - n_data_TPD_each_fold * (K - 1)  # Last fold will hold the rest of the data

    if data_div_option == 2:
        # -- Based on K-fold partition with original ratio of TPDs and FPDs ---
        FPD_TPD_ratio = n_data_FPD / n_data_TPD

        n_data_FPD_each_fold = n_data_FPD // K
        n_data_FPD_last_fold = n_data_FPD - n_data_FPD_each_fold * (K - 1)  # Last fold will hold the rest of the data

    elif data_div_option == 3:
        FPD_TPD_ratio = custom_FPD_TPD_ratio

        n_data_FPD_each_fold = np.floor(FPD_TPD_ratio * n_data_TPD_each_fold)  # Each fold will have this number of FPDs
        n_data_FPD_last_fold = np.floor(FPD_TPD_ratio * n_data_TPD_last_fold)  # Last fold will hold the rest of the data

    else:
        print('\nWrong data division option chosen.\n')
        return

    # Creating K-fold data
    X_K_fold = [None] * K  # List variable to hold the each data fold
    Y_K_fold = [None] * K

    for i in range(1, K):
        start_idx_TPD = (i - 1) * n_data_TPD_each_fold
        end_idx_TPD = i * n_data_TPD_each_fold

        start_idx_FPD = (i - 1) * n_data_FPD_each_fold
        end_idx_FPD = i * n_data_FPD_each_fold

        X_K_fold[i - 1] = np.vstack((X_TPD_norm_rand[start_idx_TPD:end_idx_TPD, :],
                                     X_FPD_norm_rand[start_idx_FPD:end_idx_FPD, :]))
        Y_K_fold[i - 1] = np.concatenate((np.ones(end_idx_TPD - start_idx_TPD),
                                          np.zeros(end_idx_FPD - start_idx_FPD)))
        
    #For Last fold
    start_idx_TPD_last = (K - 1) * n_data_TPD_each_fold
    end_idx_TPD_last = (K - 1) * n_data_TPD_each_fold + n_data_TPD_last_fold

    start_idx_FPD_last = (K - 1) * n_data_FPD_each_fold
    end_idx_FPD_last = (K - 1) * n_data_FPD_each_fold + n_data_FPD_last_fold

    X_K_fold[K - 1] = np.vstack((X_TPD_norm_rand[start_idx_TPD_last:end_idx_TPD_last, :],
                                 X_FPD_norm_rand[start_idx_FPD_last:end_idx_FPD_last, :]))
    Y_K_fold[K - 1] = np.concatenate((np.ones(end_idx_TPD_last - start_idx_TPD_last),
                                      np.zeros(end_idx_FPD_last - start_idx_FPD_last)))

    return X_K_fold, Y_K_fold, n_data_TPD_each_fold, n_data_TPD_last_fold, n_data_FPD_each_fold, n_data_FPD_last_fold, FPD_TPD_ratio, rand_num_TPD, rand_num_FPD


def divide_by_participants(data_file_names, extracted_data, FPD_extracted, X_TPD_norm, X_FPD_norm):
    # Finding the number of data files for different participants
    n_data_files = len(TPD_extracted)
    n_DF_P1, n_DF_P2, n_DF_P3, n_DF_P4, n_DF_P5 = 0, 0, 0, 0, 0
    
    for i in range(n_data_files):
        participant_num = int(data_file_names[i][1])
        if participant_num == 1:
            n_DF_P1 += 1
        elif participant_num == 2:
            n_DF_P2 += 1
        elif participant_num == 3:
            n_DF_P3 += 1
        elif participant_num == 4:
            n_DF_P4 += 1
        elif participant_num == 5:
            n_DF_P5 += 1

    # Division by participant
    n_TPD_P1, n_TPD_P2, n_TPD_P3, n_TPD_P4, n_TPD_P5 = 0, 0, 0, 0, 0
    n_FPD_P1, n_FPD_P2, n_FPD_P3, n_FPD_P4, n_FPD_P5 = 0, 0, 0, 0, 0

    for i in range(n_DF_P1):
        n_TPD_P1 += len(TPD_extracted[i])
        n_FPD_P1 += len(FPD_extracted[i])

    for i in range(n_DF_P1, n_DF_P1 + n_DF_P2):
        n_TPD_P2 += len(TPD_extracted[i])
        n_FPD_P2 += len(FPD_extracted[i])

    for i in range(n_DF_P1 + n_DF_P2, n_DF_P1 + n_DF_P2 + n_DF_P3):
        n_TPD_P3 += len(TPD_extracted[i])
        n_FPD_P3 += len(FPD_extracted[i])

    for i in range(n_DF_P1 + n_DF_P2 + n_DF_P3, n_DF_P1 + n_DF_P2 + n_DF_P3 + n_DF_P4):
        n_TPD_P4 += len(TPD_extracted[i])
        n_FPD_P4 += len(FPD_extracted[i])

    for i in range(n_DF_P1 + n_DF_P2 + n_DF_P3 + n_DF_P4, n_DF_P1 + n_DF_P2 + n_DF_P3 + n_DF_P4 + n_DF_P5):
        n_TPD_P5 += len(TPD_extracted[i])
        n_FPD_P5 += len(FPD_extracted[i])

    X_TPD_by_participant = [X_TPD_norm[:n_TPD_P1, :], X_TPD_norm[n_TPD_P1:n_TPD_P1 + n_TPD_P2, :],
                            X_TPD_norm[n_TPD_P1 + n_TPD_P2:n_TPD_P1 + n_TPD_P2 + n_TPD_P3, :],
                            X_TPD_norm[n_TPD_P1 + n_TPD_P2 + n_TPD_P3:n_TPD_P1 + n_TPD_P2 + n_TPD_P3 + n_TPD_P4, :],
                            X_TPD_norm[n_TPD_P1 + n_TPD_P2 + n_TPD_P3 + n_TPD_P4:n_TPD_P1 + n_TPD_P2 + n_TPD_P3 + n_TPD_P4 + n_TPD_P5, :]]

    X_FPD_by_participant = [X_FPD_norm[:n_FPD_P1, :], X_FPD_norm[n_FPD_P1:n_FPD_P1 + n_FPD_P2, :],
                            X_FPD_norm[n_FPD_P1 + n_FPD_P2:n_FPD_P1 + n_FPD_P2 + n_FPD_P3, :],
                            X_FPD_norm[n_FPD_P1 + n_FPD_P2 + n_FPD_P3:n_FPD_P1 + n_FPD_P2 + n_FPD_P3 + n_FPD_P4, :],
                            X_FPD_norm[n_FPD_P1 + n_FPD_P2 + n_FPD_P3 + n_FPD_P4:n_FPD_P1 + n_FPD_P2 + n_FPD_P3 + n_FPD_P4 + n_FPD_P5, :]]

    return X_TPD_by_participant, X_FPD_by_participant





def get_prediction_accuracies(model, Z_train, Y_train, Z_test, Y_test, n_test_data_TPD, n_test_data_FPD):
    # Evaluation of training and testing errors
    train_prediction_overall = model.predict(Z_train)
    train_accuracy_overall = accuracy_score(Y_train, train_prediction_overall)

    test_prediction_overall = model.predict(Z_test)
    test_accuracy_overall = accuracy_score(Y_test, test_prediction_overall)

    test_accuracy_TPD = accuracy_score(Y_test[:n_test_data_TPD], test_prediction_overall[:n_test_data_TPD])
    test_accuracy_FPD = accuracy_score(Y_test[n_test_data_TPD:], test_prediction_overall[n_test_data_TPD:])

    return train_accuracy_overall, test_accuracy_overall, test_accuracy_TPD, test_accuracy_FPD


def sigmoid(z):
    # Compute the sigmoid of z
    return 1 / (1 + np.exp(-z))


def projectData(X, U, K):
    # PROJECTDATA Computes the reduced data representation when projecting only 
    # on to the top k eigenvectors
    #    Z = projectData(X, U, K) computes the projection of 
    #    the normalized inputs X into the reduced dimensional space spanned by
    #    the first K columns of U. It returns the projected examples in Z.
    
    # % ====================== YOUR CODE HERE ======================
    # % Instructions: Compute the projection of the data using only the top K 
    # %               eigenvectors in U (first K columns). 
    # %               For the i-th example X(i,:), the projection on to the k-th 
    # %               eigenvector is given as follows:
    # %                    x = X(i, :)';
    # %                    projection_k = x' * U(:, k);
    # %



    # Compute the projection of the data using the top K eigenvectors in U
    U_reduced = U[:, :K]
    Z = np.dot(X, U_reduced)
    return Z


def get_overall_test_prediction(final_test_prediction, n_TPD, n_TPD_each_fold, n_TPD_last_fold,
                                n_FPD, n_FPD_each_fold, n_FPD_last_fold, n_iter, rand_num_TPD, rand_num_FPD):
    test_TPD_prediction_rand = np.zeros(n_TPD) # Will hold the randomized TPD predictions from testing
    test_FPD_prediction_rand = np.zeros(n_FPD)
    
    for i in range(n_iter-1):
        test_TPD_prediction_rand[(i*n_TPD_each_fold) : ( (i+1)*n_TPD_each_fold )] = final_test_prediction[i][:n_TPD_each_fold]
        test_FPD_prediction_rand[(i*n_FPD_each_fold) : ( (i+1)*n_FPD_each_fold )] = final_test_prediction[i][n_TPD_each_fold:]

    test_TPD_prediction_rand[(n_iter-1)*n_TPD_each_fold:] = final_test_prediction[n_iter-1][:n_TPD_last_fold]
    test_FPD_prediction_rand[(n_iter-1)*n_FPD_each_fold:] = final_test_prediction[n_iter-1][n_TPD_last_fold:]

    test_TPD_prediction = np.zeros(n_TPD) # Will hold the non-randomized TPD predictions from testing
    test_FPD_prediction = np.zeros(n_FPD)

    for i in range(n_TPD):
        test_TPD_prediction[rand_num_TPD[i]] = test_TPD_prediction_rand[i]

    for i in range(n_FPD):
        test_FPD_prediction[rand_num_FPD[i]] = test_FPD_prediction_rand[i]

    return test_TPD_prediction, test_FPD_prediction



def map_ML_detections(sensor_data_sgmntd_cmbd_all_sensors_labeled, M_sntn_map, n_label,
                       prediction_current_dataset_TPD, prediction_current_dataset_FPD,
                       matching_index_TPD, matching_index_FPD):
    sensor_data_sgmntd_cmbd_all_sensors_ML = np.zeros(sensor_data_sgmntd_cmbd_all_sensors_labeled.shape)

    if n_label:  # When there is a detection by the sensor system
        for k in range(1, n_label + 1):
            L_min = np.where(sensor_data_sgmntd_cmbd_all_sensors_labeled == k)[0][0]  # Sample no. corresponding to the start of the label
            L_max = np.where(sensor_data_sgmntd_cmbd_all_sensors_labeled == k)[0][-1]  # Sample no. corresponding to the end of the label
            indv_window = np.zeros(len(M_sntn_map))  # This window represents the current detection, which will be compared with M_sntn_map to find if it is TPD or FPD
            indv_window[L_min:L_max + 1] = 1
            overlap = np.sum(indv_window * M_sntn_map)  # Checks the overlap with the maternal sensation

            if overlap:
                # This is a TPD
                if prediction_current_dataset_TPD[matching_index_TPD] == 1:  # Checks the detection from the classifier
                    sensor_data_sgmntd_cmbd_all_sensors_ML[L_min:L_max + 1] = 1
                matching_index_TPD += 1
            else:
                # This is an FPD
                if prediction_current_dataset_FPD[matching_index_FPD] == 1:  # Checks the detection from the classifier
                    sensor_data_sgmntd_cmbd_all_sensors_ML[L_min:L_max + 1] = 1
                matching_index_FPD += 1

    return sensor_data_sgmntd_cmbd_all_sensors_ML, matching_index_TPD, matching_index_FPD


def define_model(n_input, n_hidden_layer, n_unit_per_layer):

    # define model
    model = Sequential()

    # define the 1st hidden layers
    model.add(Dense(n_unit_per_layer, input_shape=(n_input,), activation='relu',
                    kernel_initializer='he_uniform'))

    # define additional hidden layers
    if n_hidden_layer > 1:
        for i in range(n_hidden_layer-1):
            model.add(Dense(n_unit_per_layer, activation='relu'))

    # define the final output layer
    model.add(Dense(1, activation='linear'))

    # define loss and optimizer
    model.compile(loss=BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    # Putting 'from_logit = True' makes the algorithm to reduce round off error.
    # While 'activation = 'linear'' was given for the output layer, the
    # 'from_logit = True' makes the actual activation of the output layer as sigmoid

    # return the model
    return model