# -*- coding: utf-8 -*-
import os
import numpy as np
import pywt
from tqdm import tqdm  

from scipy.fft import fft
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score
from scipy.signal import convolve, stft, correlate2d, hilbert, spectrogram
from scipy.integrate import simps

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import regularizers
from SP_Functions import get_IMU_map,get_segmented_data, getParticipantID

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')


def extract_detections(M_sntn_map, sensor_data_fltd, sensor_data_sgmntd, sensor_data_sgmntd_cmbd_all_sensors_labeled, n_label, n_FM_sensors,
                       IMU_aclm_fltd, IMU_rotation_fltd_1D):
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
    
    current_file_aclm_TPD = []
    current_file_aclm_FPD = []
    current_file_rot_TPD = []
    current_file_rot_FPD = []
    
    
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
            
            current_file_aclm_TPD.append(IMU_aclm_fltd[L_min:L_max + 1])
            current_file_rot_TPD.append(IMU_rotation_fltd_1D[L_min:L_max + 1])
            
        else:
            k_FPD += 1
            current_FPD_extraction = np.zeros((L_max - L_min + 1, n_FM_sensors))
            for j in range(n_FM_sensors):
                current_FPD_extraction[:, j] = sensor_data_fltd[j][L_min:L_max + 1]
                
            current_file_FPD_extraction_cell.append(current_FPD_extraction)
            current_file_extracted_FPD_weightage.append(tmp_var)
            
            current_file_aclm_FPD.append(IMU_aclm_fltd[L_min:L_max + 1])
            current_file_rot_FPD.append(IMU_rotation_fltd_1D[L_min:L_max + 1])
            
    return (
        current_file_TPD_extraction_cell,
        current_file_FPD_extraction_cell,
        current_file_extracted_TPD_weightage,
        current_file_extracted_FPD_weightage,
        
        current_file_aclm_TPD, current_file_rot_TPD, current_file_aclm_FPD, current_file_rot_FPD
        
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


def getTFMF(S):
    """
    Author: @ Omar Ibn Shahid
    This function doesn't work properly due to the incorrect intialization of referenceTemplate. It gives 0. So everything turn to 0
    Parameters
    ----------
    S : TYPE
        DESCRIPTION.

    Returns
    -------
    Output: TYPE
            DESCRIPTION.

    """
    # Set parameters
    windowSize = 256
    hopSize = 128
    timeOffset = 50
    frequencyOffset = 10

    # Compute the STFT of the signal  # Shape of f will be (n_freq,) ||| # Shape of t will be (n_time,)
    f, t, S_tf = spectrogram(S, window='hann', nperseg=windowSize, noverlap=windowSize - hopSize)

    # Define the reference template equation with time offset
    referenceTemplate = 2**(1/4) * np.exp(-np.pi * ((t[:, np.newaxis] - timeOffset)**2)) * np.exp(-1j * 2 * np.pi * f[np.newaxis, :] * frequencyOffset)
    # Perform TFMF with time offset
    correlation = np.sum(np.sum(np.dot(S_tf,np.conj(referenceTemplate)))) # This works. but it all gives the same value near to 0
    
    output = np.abs(correlation)
    return output

def getConvolvedSignal(data):
    """
    Author: @ Monaf Chowdhury
    Performs 1D convolution on the input data using a Gaussian filter kernel. It convolves the input data with the kernel and 
    returns the convolved signal.
    
    Parameters
    ----------
    data : numpy.ndaarray. Shape(Number of elements)
        Data is value for a single segment and single sensor

    Returns
    -------
    convolved_signal: numpy.ndarray. Shape(Number of elements)
        Convolution of data
    """
    # Define the Gaussian filter kernel
    sigma = 1  # Standard deviation
    filter_length = 7  # Filter length (odd number)

    # Create a time vector for the filter kernel
    t = np.arange(-(filter_length-1)/2, (filter_length-1)/2 + 1)

    # Calculate the Gaussian filter values for the time vector
    F = np.exp(-t**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

    # Perform 1D convolution using the convolve function
    convolved_signal = convolve(data, F, mode='same')
    return convolved_signal

def get1Dconv(convolved_signal, key):
    """
    Author:@ Omar Ibn Rashid
    Performs different types of key mapping and feature extraction on the convolved signal data
    Parameters
    ----------
    convolved_signal : numpy.ndarray. Shape(Number of elements)
        Convolution of data
    key : int
        Integer value ranging from 1 to 21 for key mapping
    Returns
    -------
    output : float
        Floating type value generated by the specific type of operation done on the convolved signal data.

    """

    if key == 1:
        output = np.mean(convolved_signal)
    elif key == 2:
        output = np.std(convolved_signal)
    elif key == 3:
        output = np.max(convolved_signal)
    elif key == 4:
        output = np.min(convolved_signal)
    elif key == 5:
        output = np.ptp(convolved_signal)                                       # Peak to peak
    elif key == 6:
        output = np.max(np.abs(convolved_signal))                               # Max of absolute value
    elif key == 7:
        output = np.sum(convolved_signal**2)
        
    elif key == 8:      
        zero_crossings = np.where(np.diff(np.sign(convolved_signal)))[0]        # Zero-Crossing Rate
        output = len(zero_crossings) / len(convolved_signal)
    elif key == 9:
        output = np.mean(np.gradient(convolved_signal))                         # Slope/Gradient
    elif key == 10:
        output = np.max(convolved_signal) - np.min(convolved_signal)            # Peak-to-Bottom difference
    elif key == 11:
        output = skew(convolved_signal)                                         # Skewness
    elif key == 12:
        output = kurtosis(convolved_signal)                                     # Kurtosis
    elif key == 13:
        output = np.sqrt(np.mean(convolved_signal**2))                          # Root Mean Square
    elif key == 14:
        output = np.percentile(convolved_signal, 75) - np.percentile(convolved_signal, 25)  # Interquartile Range
        
    elif key == 15:
        output = np.mean(np.abs(convolved_signal - np.mean(convolved_signal)))  # Mean Absolute Deviation
    elif key == 16:
        output = np.mean(np.abs(convolved_signal))                              # Absolute Mean
    elif key == 17:
        output = np.sum(convolved_signal[convolved_signal > 0])                 # Positive Area
    elif key == 18:
        output = np.sum(convolved_signal[convolved_signal < 0])                 # Negative Area
    elif key == 19:
        mean_diff = convolved_signal - np.mean(convolved_signal)
        mean_crossings = np.where(np.diff(np.sign(mean_diff)))[0]
        output = len(mean_crossings) / len(convolved_signal)                    # Mean Crossing Rate
    elif key == 20:
        # Monaf's implementation
        unique, counts = np.unique(convolved_signal.flatten(), return_counts=True) # Flatten the array and count occurrences of each unique value
        probs = counts / np.sum(counts)                                         # Compute probabilities
        output = -np.sum(probs * np.log2(probs))                                # Compute entropy
    elif key == 21: 
        peak_value = np.max(np.abs(convolved_signal))
        rms_value = np.sqrt(np.mean(convolved_signal**2))
        output = peak_value / rms_value                                         # Crest Factor
    else:
        print('Error in calling 1D convolution function')
        output = None
    return output

def getWaveCoff(data):
    """
    Author: @Omar Ibn Rashid
    Calculates the wavelet coefficients of the input data using a specified wavelet transformation. Wavelet coefficients represent 
    the decomposition of the signal in terms of different wavelet functions.
    
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    AC : TYPE
        The wavelet coefficients.

    """
    wavelet_name = 'haar'   # Name of the wavelet to be used for transformation.
    AC, _ = pywt.dwt(data, wavelet_name)   # Perform discrete wavelet transform

    # 'AC' contains the wavelet coefficients, and the ignored output is typically the approximation coefficients. 
    # One can use 'AC' for further analysis or processing.
    return AC

def getInstFreq(data, Fs_sensor):
    """
    Author: @ Omar Ibn Shahid
    Calculates the instantaneous frequency of a given signal using the Hilbert transform. Instantaneous frequency provides information 
    about how the frequency content of the signal changes over time
    
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    Fs_sensor : int
        Sampling rate of the signal. Usually it is set to 1024.

    Returns
    -------
    instFreq : TYPE
        Instantaneous frequency of the signal..

    """

    # Apply the Hilbert transform to the input signal to obtain the analytic signal 
    # analytic signal = which is a complex-valued signal representing the original signal's amplitude and phase information
    hx = hilbert(data)
    
    # Calculate the instantaneous phase of the analytic signal and unwrap it to avoid phase wrapping issues
    phase = np.unwrap(np.angle(hx))
    
    # Calculate the instantaneous frequency by taking the derivative of the phase and dividing by the sampling frequency of the sensor
    instFreq = np.diff(phase) * Fs_sensor / (2 * np.pi) # division by 2π is required to convert the phase difference (in radians) into frequency
    
    # The result is a time-varying estimate of the instantaneous frequency of the signal, which provides insights 
    # into how the frequency content changes over time.
    
    return instFreq

def getInstAmp(data):
    """
    Author: @ Omar Ibn Shahid
    Calculates the instantaneous amplitude of a given signal using the Hilbert transform. Instantaneous amplitude represents 
    the magnitude of the analytic signal and provides information about signal envelope variations.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    inst_Amplitude : TYPE
        Instantaneous amplitude of the signal.

    """
    # Apply the Hilbert transform to the input signal to obtain the analytic signal
    hx = hilbert(data)

    # Calculate the instantaneous amplitude as the absolute value of the analytic signal
    inst_Amplitude = np.abs(hx)

    # The result is a time-varying estimate of the instantaneous amplitude of
    # the signal, which provides insights into how the signal envelope changes over time.
    return inst_Amplitude

def getMorphological(signal, Fs_sensor):
    """
    Author: @ Abdul Monaf Chowdhury 
    Function provides the morphological properties of the signal. i.e, Absolute area, relative area, and absolute area of the differential
    
        Absolute Area: The absolute area of a signal is calculated by summing the absolute values of all samples in the signal array. 
        Relative Area: The relative area of a signal is calculated by dividing the absolute area of the signal by 
                        the total number of samples in the signal array. 
        Absolute Area of the Differential: The absolute area of the differential (or cumulative sum) of a signal is calculated by 
                        first computing the differential of the signal (the difference between consecutive samples) 
                        and then finding the absolute area of the resulting signal.

    Parameters
    ----------
    signal : numpy.ndarray. Shape(Number of samples,)
        DESCRIPTION.
    Fs_sensor: int. 
        Sampling rate of the signal
    Returns
    -------
    absolute_area : numpy.float64
        Absolute area of the signal
    relative_area : numpy.float64
        Relative area of the signal
    absolute_area_differential : numpy.float64
        Absolute Area of the Differential of the signal            
    
    """
    time_interval = 1 / Fs_sensor
    total_samples = len(signal)
    
    # Calculate absolute area
    absolute_area = simps(np.abs(signal), dx=time_interval)
    
    # Calculate differential
    differential = np.diff(signal) # Taking numerical derivative
    
    # Calculate absolute area of the differential
    absolute_area_differential = simps(np.abs(differential), dx=time_interval)
    
    # Calculate relative area
    relative_area = absolute_area_differential / absolute_area * 100
    # relative_area = absolute_area / total_samples
    # relative_area = np.sum(signal) * time_interval
    
    return absolute_area, relative_area, absolute_area_differential


def extract_features(TPD_extracted, extracted_TPD_weightage, FPD_extracted, extracted_FPD_weightage, threshold, Fs_sensor, n_FM_sensors,
                     IMU_aclm_TPD, IMU_aclm_FPD, IMU_rot_TPD, IMU_rot_FPD):
    """
    Algorithm: @ Abhishek Kumar Ghosh
    Author: @ Monaf Chowdhury, Moniruzzaman Akash, Omar Ibn Shahid 
    Extracts features of TPD and FPD detections. Usually 16 features per sensor are detected. So 16*6 = 96 features
    Also detection length is also passed as a feature. So in total, feature = 16*6+1 = 97 features
    Parameters
    ----------
    TPD_extracted : List. Shape (number of files, number of TPD detections, detection length, number of FM sensor)
        Extracted TPD of all of files are passed here. 
    extracted_TPD_weightage :  List . Shape (number of files, number of TPD detection)
        Number of common sensors for each detection.
    FPD_extracted : List. Shape (number of files, number of FPD detections, detection length, number of FM sensor)
        Extracted FPD of all of files are passed here. 
    extracted_FPD_weightage : List . Shape (number of files, number of FPD detection)
        
    threshold : numpy.ndarray. Shape(number of files, number of FM Sensor)
        Threshold values are usually float and used to extract some of the features.
    Fs_sensor : int
        The sampling rate of the sensor data. Usual case scenario Fs_sensor = 1024
    n_FM_sensors : int
        Number of sensors based on which this segmenation process was done. Ideally it has value of 6.

    Returns
    -------
    X_TPD : numpy.ndarray. shape(total number of TPD detected, total number of features)
        Extracted features of TPD Based on some defined policy. Based on this models are trained. 
    X_FPD : numpy.ndarray. shape(total number of FPD detected, total number of features)
        Extracted features of TPD Based on some defined policy. Based on this models are trained. 
    n_TPD : int
        Total number of TPD detected for all the files. 
    n_FPD : int
        Total number of FPD detected for all the files.
    total_duration_TPD : int 
        Duration of TPD in hours.
    total_duration_FPD : int 
        Duration of FPD in hours.

    """
    
    # Total duration of TPDs and FPDs
    n_TPD = 0  # Total no. of TPD segment in all datafile selected
                # This will be less than TPD from the thresholding algorithm as each one may hold multiple maternal sensation detection
    n_FPD = 0  # Total no. of FPD segment in all datafile selected
    total_TPD_data = 0  # Total TP datapoints in a data file
    total_FPD_data = 0  # Total FP datapoints in a data file

    for i in range(len(TPD_extracted)): # For each data file
        n_TPD += len(TPD_extracted[i])
        n_FPD += len(FPD_extracted[i])

        for j in range(len(TPD_extracted[i])): # For each segment count how many TP data points
            total_TPD_data += len(TPD_extracted[i][j])

        for j in range(len(FPD_extracted[i])):
            total_FPD_data += len(FPD_extracted[i][j]) # For each segment count how many FP data points

    total_duration_TPD = total_TPD_data/Fs_sensor/3600  # Duration of TPD in hrs
    total_duration_FPD = total_FPD_data/Fs_sensor/3600  # Duration of FPD in hrs
    
    n_common_features = 53 #i.e. max, mean, sum, sd, percentile, skew, kurtosis, .....
    total_features = (n_FM_sensors+2) * n_common_features + 1 #Extra one feature for segment duration
    
    
    # Extraction of features from TPDs and FPDs
    X_TPD = np.zeros((n_TPD, total_features))  # Features of TPD
    X_FPD = np.zeros((n_FPD, total_features))  # Feature of FPD
    index_TPD = 0
    index_FPD = 0

    print("Extracting features for TPD...")
    # Initialize tqdm progress bar
    total_iterations = sum(len(TPD_extracted[i]) for i in range(len(TPD_extracted)))
    pbar = tqdm(total=total_iterations, desc='Processing TPD segments')

    for i in range(len(TPD_extracted)): # For each data file
        for j in range(len(TPD_extracted[i])): # For each TP segment in a data file
            
            #X_TPD[index_TPD,1] = data_file_GA[i]
            #X_TPD[index_TPD,2] = extracted_TPD_weightage[i][j]#  Duration of each TPD in s
            X_TPD[index_TPD, 0] = len(TPD_extracted[i][j]) / Fs_sensor  # Duration of each TPD in s

            for k in range(n_FM_sensors+2):
                
                if k == n_FM_sensors:  # IMU_aclm_TPD, IMU_rot_TPD
                    S = IMU_rot_TPD[i][j]
                    S_thd = S
                    S_thd_above = S
                
                elif k == n_FM_sensors+1: 
                    S = IMU_aclm_TPD[i][j]
                    S_thd = S
                    S_thd_above = S

                else: 
                    S = TPD_extracted[i][j][:, k]
                    S_thd = np.abs(S) - threshold[i, k]
                    S_thd_above = S_thd[S_thd > 0]

    # =============================================================================
                    # S_MPDatom = np.abs(TPD_extracted[i][j]) - threshold[i] # get getMPDatom expects 2D array. 
    # =============================================================================
                    # 2D array representing a set of epoch signals. Each row of the array corresponds to an epoch signal, 
                    # and each column corresponds to a sample in the signal.
                
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
                
# =============================================================================
#                 New features added by Omar 
# =============================================================================
                # Calculating instantaneous frequency and amplitude
                X_TPD[index_TPD, k * n_common_features + 17] = np.mean(getInstAmp(S_thd))
                X_TPD[index_TPD, k * n_common_features + 18] = np.mean(getInstFreq(S, Fs_sensor))

                X_TPD[index_TPD, k * n_common_features + 19] = np.std(getInstAmp(S_thd))
                X_TPD[index_TPD, k * n_common_features + 20] = np.std(getInstFreq(S, Fs_sensor))

                X_TPD[index_TPD, k * n_common_features + 21] = np.max(getInstAmp(S_thd)) - np.min(getInstAmp(S_thd))
                X_TPD[index_TPD, k * n_common_features + 22] = np.max(getInstFreq(S, Fs_sensor)) - np.min(getInstFreq(S, Fs_sensor))

                # Calculating Wavelet Coefficient
                X_TPD[index_TPD, k * n_common_features + 23] = np.mean(getWaveCoff(S))
                X_TPD[index_TPD, k * n_common_features + 24] = np.median(getWaveCoff(S))
                X_TPD[index_TPD, k * n_common_features + 25] = np.std(getWaveCoff(S))      
                
                # Calculate 1D Convolution -> Feature count starts at 26 and ends at 46
                convolved_signal = getConvolvedSignal(S_thd)
                for conv_index in range(21):
                    X_TPD[index_TPD, k * n_common_features + 26 + conv_index] = get1Dconv(convolved_signal, conv_index + 1)
                    
                # Calculate number of atoms
                X_TPD[index_TPD, k * n_common_features + 47] = 1 # getMPDatom(S_MPDatom) #  Passing S_MPDatom (2D array) 
                
                # Calculate time frequency matched filter
                X_TPD[index_TPD, k * n_common_features + 48] = getTFMF(S_thd)
                
# =============================================================================
#                 New features added by Monaf 
# =============================================================================
                X_TPD[index_TPD, k * n_common_features + 49] = np.min(S_thd)  # Min value
                X_TPD[index_TPD, k * n_common_features + 50] = np.median(S_thd)  # Median value
                
                # Calculate the morphological features. 
                absolute_area, relative_area, absolute_area_differential = getMorphological(S_thd, Fs_sensor)
                X_TPD[index_TPD, k * n_common_features + 51] = absolute_area
                X_TPD[index_TPD, k * n_common_features + 52] = relative_area
                X_TPD[index_TPD, k * n_common_features + 53] = absolute_area_differential
            
            pbar.update(1)
            index_TPD += 1
            
    # Close tqdm progress bar
    pbar.close()

    
    print("Extracting features for FPD...")
    # Initialize tqdm progress bar
    total_iterations = sum(len(FPD_extracted[i]) for i in range(len(FPD_extracted)))
    pbar = tqdm(total=total_iterations, desc='Processing FPD segments')    

    for i in range(len(FPD_extracted)):
        for j in range(len(FPD_extracted[i])):
            
            #       X_FPD[index_FPD,1] = data_file_GA[i]
            #       X_FPD[index_FPD,2] = extracted_FPD_weightage[i][j] # Duration of each TPD in s
            
            X_FPD[index_FPD, 0] = len(FPD_extracted[i][j]) / Fs_sensor  # Duration in s
            
            
            # Feature extraction for left accelerometer
            for k in range(n_FM_sensors+2):               
                if k == n_FM_sensors:  # IMU_aclm_FPD, IMU_rot_FPD
                    S = IMU_rot_FPD[i][j]
                    S_thd = S
                    S_thd_above = S
                
                elif k == n_FM_sensors+1 :
                    S = IMU_aclm_FPD[i][j]
                    S_thd = S
                    S_thd_above = S
                
                else:
                    S = FPD_extracted[i][j][:, k]
                    S_thd = np.abs(S) - threshold[i, k]
                    S_thd_above = S_thd[S_thd > 0]
                    
                    # =============================================================================
                    # S_MPDatom = np.abs(FPD_extracted[i][j]) - threshold[i] # get getMPDatom expects 2D array. 
                    # =============================================================================
                
                
                # Statistical features
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
                
                # Frequency domain features
                _, _, X_FPD[index_FPD, k * n_common_features + 11] = get_frequency_mode(S, Fs_sensor, 1)  # Gives the main frequency mode above 1 Hz
                X_FPD[index_FPD, k * n_common_features + 12] = getPSD(S, Fs_sensor, 1, 2)
                X_FPD[index_FPD, k * n_common_features + 13] = getPSD(S, Fs_sensor, 2, 5)
                X_FPD[index_FPD, k * n_common_features + 14] = getPSD(S, Fs_sensor, 5, 10)
                X_FPD[index_FPD, k * n_common_features + 15] = getPSD(S, Fs_sensor, 10, 20)
                X_FPD[index_FPD, k * n_common_features + 16] = getPSD(S, Fs_sensor, 20, 30)
                
                # Calculating instantaneous frequency and amplitude for FPD
                X_FPD[index_FPD, k * n_common_features + 17] = np.mean(getInstAmp(S_thd))
                X_FPD[index_FPD, k * n_common_features + 18] = np.mean(getInstFreq(S, Fs_sensor))

                X_FPD[index_FPD, k * n_common_features + 19] = np.std(getInstAmp(S_thd))
                X_FPD[index_FPD, k * n_common_features + 20] = np.std(getInstFreq(S, Fs_sensor))

                X_FPD[index_FPD, k * n_common_features + 21] = np.max(getInstAmp(S_thd)) - np.min(getInstAmp(S_thd))
                X_FPD[index_FPD, k * n_common_features + 22] = np.max(getInstFreq(S, Fs_sensor)) - np.min(getInstFreq(S, Fs_sensor))

                # Calculating Wavelet Coefficient for FPD
                X_FPD[index_FPD, k * n_common_features + 23] = np.mean(getWaveCoff(S))
                X_FPD[index_FPD, k * n_common_features + 24] = np.median(getWaveCoff(S))
                X_FPD[index_FPD, k * n_common_features + 25] = np.std(getWaveCoff(S))

                # Calculate 1D Convolution for FPD -> Feature count starts at 26 and ends at 46
                convolved_signal = getConvolvedSignal(S_thd)
                for conv_index in range(21):
                    X_FPD[index_FPD, k * n_common_features + 26 + conv_index] = get1Dconv(convolved_signal, conv_index + 1)


                # Calculate number of atoms for FPD
                X_FPD[index_FPD, k * n_common_features + 47] = 1 # getMPDatom(S_MPDatom)  # Passing S_MPDatom (2D array)
      
                # Calculate time frequency matched filter for FPD
                X_FPD[index_FPD, k * n_common_features + 48] = getTFMF(S_thd)     
                
                X_FPD[index_FPD, k * n_common_features + 49] = np.min(S_thd)  # Min value
                X_FPD[index_FPD, k * n_common_features + 50] = np.median(S_thd)  # Median value
                # Calculate the morphological features. 
                absolute_area, relative_area, absolute_area_differential = getMorphological(S_thd, Fs_sensor)
                X_FPD[index_FPD, k * n_common_features + 51] = absolute_area
                X_FPD[index_FPD, k * n_common_features + 52] = relative_area
                X_FPD[index_FPD, k * n_common_features + 53] = absolute_area_differential
                

            index_FPD += 1
            pbar.update(1)
            
    # Close tqdm progress bar
    pbar.close()

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
    """
    Algorithm: @ Abhishek Kumar Ghosh
    Author: @ Moniruzzaman Akash
    Mean normalization occurs here. 
    Parameters
    ----------
    X : numpy.ndarray. Shape (total number of TPD and FPD combined, total number of features)
        X is passed after feature extraction. So, TPD and FPD are vertically combined and has the above mentioned shape 
    Returns
    -------
    X_norm : numpy.ndarray. Shape (total number of TPD and FPD combined, total number of features)
        Typically the values will be between -1 to 1 .
    mu : numpy.ndarray. Shape (total number of features)
        It will create a vector of 'total number of features' elements and values will contain the mean value for each feature
    dev : numpy.ndarray. Shape (total number of features)
        It will create a vector of 'total number of features' elements and values will contain the deviation/span value for each feature.

    """
    
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
    Algorithm: @ Abhishek Kumar Ghosh
    Author: @ Moniruzzaman Akash
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


def divide_by_participants(data_file_names, TPD_extracted, FPD_extracted, X_TPD_norm, X_FPD_norm):
    """
    Algorithm: @ Abhishek Kumar Ghosh
    Author: @ Moniruzzaman Akash
    Divide data by participant
    Parameters
    ----------
    data_file_names : List. Shape(number of data files)
        Data file namess in 'str' format
    TPD_extracted : List. Shape (number of files, number of TPD detections, detection length, number of FM sensor)
        Extracted TPD of all of files are passed here. 
    FPD_extracted : List. Shape (number of files, number of FPD detections, detection length, number of FM sensor)
        Extracted FPD of all of files are passed here.
    X_TPD_norm : numpy.ndarray. Shape (total number of TPD detections, total number of features)
        Typically the values will be between -1 to 1 .
    X_FPD_norm : numpy.ndarray. Shape (total number of FPD detections, total number of features)
        Typically the values will be between -1 to 1 .

    Returns
    -------
    X_TPD_by_participant : List. Shape(number of participant, number of TPD detections, number of features)
        X_TPD is split into a list of numpy.ndarrays where each element of list is for a particular participant.
    X_FPD_by_participant : List. Shape(number of participant, number of FPD detections, number of features)
        X_FPD is split into a list of numpy.ndarrays where each element of list is for a particular participant.

    """
    # Finding the number of data files for different participants
    n_data_files = len(TPD_extracted)
    n_DF_P1, n_DF_P2, n_DF_P3, n_DF_P4, n_DF_P5 = 0, 0, 0, 0, 0
    
    for i in range(n_data_files):
        participant_num = int(getParticipantID(data_file_names[i]))
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
    """
    Algorithm: @ Abhishek Kumar Ghosh
    Author: @ Moniruzzaman Akash
    Generates the prediction accuracy of the trained model 
    Parameters
    ----------
    model : Model that was trained at the point in iteration
        
    Z_train : numpy.ndarray(number of values, number of features)
        Trained data passed as 2D array for prediction
    Y_train : numpy.ndarray. Shape (number of values)
        Trained data labels
    Z_test : numpy.ndarray(number of values, number of features)
        Test data passed as 2D array for evaluation
    Y_test : numpy.ndarray. Shape (number of values)
        Test data labels
    n_test_data_TPD : int
        Number of TPD in the current y_test.
    n_test_data_FPD : int
        Number of FPD in the current y_test.

    Returns
    -------
    train_accuracy_overall : int
        Float value of training accuracy for that particular iteration.
    test_accuracy_overall : int 
        Float value of testing accuracy for that particular iteration.
    test_accuracy_TPD : int 
        Float value of testing accuracy of detecting TPD for that particular iteration.
    test_accuracy_FPD : int 
        Float value of testing accuracy of detecting FPD for that particular iteration.

    """
    # Evaluation of training and testing errors
    train_prediction_overall = model.predict(Z_train)
    train_accuracy_overall = accuracy_score(Y_train, train_prediction_overall)

    test_prediction_overall = model.predict(Z_test)
    test_accuracy_overall = accuracy_score(Y_test, test_prediction_overall)

    test_accuracy_TPD = accuracy_score(Y_test[:n_test_data_TPD], test_prediction_overall[:n_test_data_TPD])
    test_accuracy_FPD = accuracy_score(Y_test[n_test_data_TPD:], test_prediction_overall[n_test_data_TPD:])

    return train_accuracy_overall, test_accuracy_overall, test_accuracy_TPD, test_accuracy_FPD


def sigmoid(z):
    """
    Algorithm: @ Abhishek Kumar Ghosh
    Author: @ Moniruzzaman Akash
    Generates a sigmoid activation function. The sigmoid function always returns a value between 0 and 1.
    For small values sigmoid returns a value close to zero, for large values sigmoid return close to 1.
    Sigmoid is equivalent to a 2-element softmax, where the second element is assumed to be zero. 

    Parameters
    ----------
    z : numpy.ndarray
        Input value.

    Returns
    -------
    numpy.ndarray
        sigmoid function always returns a value between 0 and 1.
    """
    # Compute the sigmoid of z
    return 1 / (1 + np.exp(-z))


def projectData(X, U, K):
    """
    Algorithm: @ Abhishek Kumar Ghosh
    Author: @ Moniruzzaman Akash
    Computes the reduced data representation when projecting only on to the top k eigenvectors
    Computes the projection of the normalized inputs X into the reduced dimensional space spanned by the first K columns of U.
    It returns the projected examples in Z.
    Compute the projection of the data using only the top K eigenvectors in U (first K columns). 

    """
    # For the i-th example X(i,:), the projection on to the k-th eigenvector is given as follows:
    # x = X(i, :)';
    # projection_k = x' * U(:, k);

    # Compute the projection of the data using the top K eigenvectors in U
    U_reduced = U[:, :K]
    Z = np.dot(X, U_reduced)
    return Z


def get_overall_test_prediction(final_test_prediction, n_TPD, n_TPD_each_fold, n_TPD_last_fold,
                                n_FPD, n_FPD_each_fold, n_FPD_last_fold, n_iter, rand_num_TPD, rand_num_FPD):
    """
    Algorithm: @ Abhishek Kumar Ghosh
    Author: @ Moniruzzaman Akash
    Presents overall prediction of TPD and FPD in the test data
    Parameters
    ----------
    final_test_prediction : List 
        
    n_TPD : int
        Number of TPD in the test data
    n_TPD_each_fold : int
        Number of TPD in each fold
    n_TPD_last_fold : int
        Number of TPD in last fold
    n_FPD : int 
        Number of FPD in the test data
    n_FPD_each_fold : int
        Number of FPD in each fold
    n_FPD_last_fold : int
        Number of FPD in last fold
    n_iter : int
        Total number of iterations
    rand_num_TPD : numpy.ndarray. Shape(number of TPD)
        Randomized index values of different TPD. 
    rand_num_FPD : numpy.ndarray. Shape(number of FPD)
        Randomized index values of different FPD. 

    Returns
    -------
    test_TPD_prediction : numpy.ndarray. Shape (number of TPD detections, 1)
        Boolean values representing prediction values in the segments of Test data.
    test_FPD_prediction : numpy.ndarray. Shape (number of FPD detections, 1)
        Boolean values representing prediction values in the segments of Test data.

    """
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
    """
    Algorithm: @ Abhishek Kumar Ghosh
    Author: @ Moniruzzaman Akash
    
    Parameters
    ----------
    sensor_data_sgmntd_cmbd_all_sensors_labeled : numpy.ndarray, Shape(sensor data values)
        Labelled values for sensations in a particular data file. 
    M_sntn_map : numpy.ndarray. Shape(sensation mapped values)
        Contains 1/0 values for that particular data file considering Sensation Data Mapping.
        A vector with all the sensation windows joined together
    n_label : int
        Total number of detections.
    prediction_current_dataset_TPD : numpy.ndarray. Shape(number of TPD detections, 1)
        Boolean values describing the predicitons 
    prediction_current_dataset_FPD : numpy.ndarray. Shape(number of FPD detections, 1)
        Boolean values describing the predicitons 
    matching_index_TPD : int
        Total TPD
    matching_index_FPD : int
        Total FPD.

    Returns
    -------
    sensor_data_sgmntd_cmbd_all_sensors_ML : List. Shape(number of files, number of values)
        
    matching_index_TPD : int
        Total TPD
    matching_index_FPD : int
        Total FPD.

    """
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


def define_model(n_input, n_hidden_layer, n_unit_per_layer, dropout_rate=0.25, l2_penalty=0.01):
    
    """
    Algorithm: @ Abhishek Kumar Ghosh
    Author: @ Monaf Chowdhury | Moniruzzaman Akash 
    
    Defines and designs the neural network model for training. It is designed using TensorFlow
    Parameters
    ----------
    n_input : int
        Gives the number of features as the input layer size.
    n_hidden_layer : int
        Number of hidden layers in the neural network architecture
    n_unit_per_layer : int
        Number of units in any given hidden layer.
    dropout_rate: float
        Dropout rate for the dropout layer
    l2_penalty: float
        L2 regularizaiton penalty

    Returns
    -------
    model : keras.engine.sequential.Sequential
        Designed neural network model for training/testing on the data.

    """

    # define model
    model = Sequential()

    # define the 1st hidden layers
    model.add(Dense(n_unit_per_layer, input_shape=(n_input,), activation='relu', kernel_initializer='he_uniform'))
    
    # Adding batch normalization and dropout layers to avoid overfitting 
    # model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    
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

    return model