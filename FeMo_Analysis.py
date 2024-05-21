# %% Import Libraries
#!/usr/bin/bash python3

import pandas as pd
import numpy as np
from tqdm import tqdm 
#import tkinter as tk
#from tkinter import filedialog
from skimage.measure import label
import glob
import os, time, sys, argparse, warnings, logging
from matplotlib import pyplot as plt

from femo import Femo# Local code imports

from skimage.measure import label
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from scipy.special import expit

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from SP_Functions import load_data_files, get_preprocessed_data, get_IMU_map, get_IMU_rot_map, get_segmented_data, \
    get_sensation_map, match_with_m_sensation, get_performance_params, get_merged_map
        
from ML_Functions import extract_detections_modified, extract_features_modified, ML_prediction


from SP_Functions import load_data_files, get_dataFile_info, get_preprocessed_data,\
    get_info_from_preprocessed_data,get_IMU_map,get_sensation_map,get_segmented_data,\
    match_with_m_sensation,get_performance_params
        
from ML_Functions import extract_detections, extract_features, normalize_features,\
    divide_by_holdout, divide_by_K_folds, divide_by_participants, get_prediction_accuracies,\
    projectData, get_overall_test_prediction, map_ML_detections, define_model
    
from Feature_Ranking import FeatureRanker
from tensorflow.keras.callbacks import EarlyStopping, Callback
# from aws_server import S3FileManager

warnings.filterwarnings('ignore')
# Change the current working directory to the directory of the current script file
os.chdir(os.path.dirname( os.path.abspath(__file__) ) )


#%% 
# model_filepath = "Model_folder/rf_model_selected_two_sensor.pkl"
data_file_path = "Data_files/F4_12_FA_8D_05_EC/log_2024_04_04_19_12_51.dat" # Data file to use when no data file argument given
# data_file_path = "Data_files/log_2024_05_09_10_44_54_IMU_thresh_chk.dat"
load_single_file = False

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='This cose uses saved model to predict FM')
# Add arguments with names
parser.add_argument('file_name', nargs="?", default=data_file_path, help='Input file path')
parser.add_argument("--server", action="store_true", help="Run on server")
parser.add_argument("--old_data", action="store_true", help="USe old data format")

# Parse the command-line arguments
args = parser.parse_args()


if args.old_data:
    data_format = '1'
    print("Using Old Belt data.")
else:
    data_format = '2'
    print("Using New Belt data.")
    
'''    
if args.server:
    S3bucket = S3FileManager('femo-sensor-logfiles')
    
    # If file name is passed use that
    if args.file_name:
        object_key = args.file_name
    else:
        #Otherwise use a predefined data
        object_key = 'DC:54:75:C2:E3:FC/log_2024_02_06_06_17_27'
    
    #Download the file
    S3bucket.download_file(object_key)
'''

# Data Format
# data_format = '2' # input("Data Format: ")


def get_file_list(folder_path, data_format):
    # Initialize an empty list to store file names
    file_list = []

    # Get list of files in the folder
    files = os.listdir(folder_path)

    # Iterate over each file
    for file_name in files:
        
        if data_format == '1':
            # Check if file ends with ".mat" extension
            if file_name.endswith(".mat"):
                # If it does, append the file name to the list
                file_list.append(folder_path + file_name)
        
        elif data_format == '2':
            # Check if file ends with ".dat" extension
            if file_name.endswith(".dat"):
                # If it does, append the file name to the list
                file_list.append(folder_path + file_name)

    return file_list

new_data_folder_path = "I:/Other computers/Desktop/WelcomeLeap/Github/FeMo_Analysis/Data_files/all_data/"
# new_data_folder_path = "D:/Monaf/New Data FM Monitoring/All_data_files_combined/"
# new_data_folder_path = "D:/Monaf/confidential_monaf_only/"
old_data_folder_path = "I:/Other computers/Desktop/WelcomeLeap/Previous Study/All subject data/Fetal movement data/"


if load_single_file:
    data_file_names = [data_file_path]
else:
    if data_format == '1':
        data_file_names = get_file_list(old_data_folder_path, data_format)
    elif data_format == '2':
        data_file_names = get_file_list(new_data_folder_path, data_format)


# %% Data Loading and Preprocessing
# data_file_path = os.path.join(os.getcwd(), data_file_path)
# print(data_file_path)


n_data_files = len(data_file_names)

#==================================================================================
#=====================    Data Loading and extraction   ===========================
#==================================================================================


##### For old data: 
##### s1 is aclm1, # s2 is aclm2, # s3 is acstc1, # s4 is acstc2, # s5 is pzplt1, # s6 is pzplt2, # s7 is flexi, # s8 is imu
##### sampling rate for all sensor is 1024

##### For new data: 
##### s1 is aclm1, # s2 is aclm2, # s3 is pzplt1, # s4 is pzplt2, # s5 is pzplt3, # s6 is pzplt4, # s7 is flexi, # s8 is imu
##### New data do not have flexi data, so we have passed a blank array for flexi data to keep the same format of return values
##### sampling rate for piezo is 1024, accelrometer 512, imu 256. We have resampled all sensor data using upsampling technique.
##### To do upsampling, we have used linear interpolation technique


# Value initialization for old data
Fs_sensor = 1024  # Frequency of sensor data sampling in Hz
Fs_sensation = 1024  # Frequency of sensation data sampling in Hz


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Load Data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Sensor description for New Belt
#                                    Original frequency    Interpolated Frequency
    #s1 > Accelerometer_1                (256 Hz)                  (1024 Hz)
    #s2 > Accelerometer_2                (256 Hz)                  (1024 Hz)
    #s3 > Piezo_1                       (1024 Hz)                 (1024 Hz)
    #s4 > Piezo_2                       (1024 Hz)                 (1024 Hz)
    #s5 > Piezo_3                       (1024 Hz)                 (1024 Hz)
    #s6 > Piezo_4                       (1024 Hz)                 (1024 Hz)
    #Flexi_force                     (Not available)                  -
    #IMU_aclm (magnitude of xyz)        (128 Hz)                 (1024 Hz)
    #IMU_rotation(euler r,p,y)          (128 Hz)                 (1024 Hz)
    #IMU_mag (magnitude of 3 magnetometers)(128 Hz)              (1024 Hz)
    
# load_data_files() function reads data files and equalize Sampling Frequency to 1024Hz for new belt
s1, s2, s3, s4, s5, s6, Flexi_data, IMU_aclm, IMU_rotation, IMU_mag, sensation_data= load_data_files(data_format, data_file_names)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pre-processing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pre-processing steps includes filtering/detrending and trimming the data
# Trimming is done to remove some initial and final unwanted data; settings inside funcction.
# Equalize data for old belt due to different size of DAC1 and DAC2
print("\n#Data pre-processing is going on...")

s1_fltd, s2_fltd, s3_fltd, s4_fltd, s5_fltd, s6_fltd, flexi_fltd, IMU_aclm_fltd, IMU_rotation_fltd, sensation_data_trimd = get_preprocessed_data(s1, s2, s3, s4, s5, s6, Flexi_data, IMU_aclm, IMU_rotation, sensation_data, Fs_sensor, data_format)
print("Preprocessing complete")

# %% SENSOR FUSION, GENERATE SEGMENTATION MAP and DATA EXTRACTION
tic = time.time()

# Parameters for segmentation and detection matching
ext_backward        = 5.0  # Backward extension length in second
ext_forward         = 5  # Forward extension length in second
FM_dilation_time    = 7 # Dilation size in seconds, its the minimum size of a fetal movement
n_FM_sensors        = 6   # Number of FM sensors
FM_min_SN           = [30,30,60,30,30,60]  # These values are selected to get SEN of 99%

#70,70,50,30,30,50


IMU_aclm_map = []
IMU_RPY_map = []
IMU_map = []

M_sntn_map =[]
threshold = np.zeros((n_data_files, n_FM_sensors))
TPD_extracted = []
FPD_extracted = []
extracted_TPD_weightage = []
extracted_FPD_weightage = []
all_data_files_schemes = []

import itertools
original_list = [30, 40, 50, 60, 70, 80, 90]
# original_list = [30, 40]
# Generate all combinations of 6 elements with replacement
all_combinations = [0]# list(itertools.product(original_list, repeat=3))
# all_combinations = [(30,30,40), (30,40,40)]

scheme_strs = ['Atleast_1_type_of_sensor', 'Atleast_2_type_of_sensor', 'Atleast_3_type_of_sensor', 'Atleast_1_sensor', 'Atleast_2_sensors', 'Atleast_3_sensors', 'Atleast_4_sensors', 'Atleast_5_sensors', 'Atleast_6_sensors']
n_schemes = len(scheme_strs)
array_shape = (n_data_files, n_schemes)
column_names = ['Scenario', 'SENSITIVITY', 'PRECISION', 'SPECIFICITY', 'ACCURACY', 'FS_SCORE', 'M_sen_before', 'M_sen_after']
result_cell = np.zeros((n_schemes*len(all_combinations), len(column_names)), dtype=object)
scenario_count = 0
overall_detections = np.zeros((n_schemes, 4 ))

TPD_indv = np.zeros(array_shape)
FPD_indv = np.zeros(array_shape)
TND_indv = np.zeros(array_shape)
FND_indv = np.zeros(array_shape)

IMU_aclm_threshold = 0.2
IMU_rot_threshold = 1 # 4
IMU_dilation_time = 4 # 4

n_Maternal_detected_movement_raw = 0
n_Maternal_detected_movement_after = 0

print("Segmentation going on...")

# ===== Loop for testing set of SN ratios =====
# for SN in all_combinations:
#     aclm_sm = SN[0]
#     pzplt_lrg_sn = SN[1]
#     pzplt_sml_sm = SN[2]
    
    
#     FM_min_SN = [aclm_sm, aclm_sm, pzplt_lrg_sn, pzplt_sml_sm, pzplt_sml_sm,  pzplt_lrg_sn]
#     print(*FM_min_SN, sep = ", ") 
#     print(f"{scenario_count+1}/{len(all_combinations)}")
    
for i in range(n_data_files):
    # Starting notification
    print('\nCurrent data file: {}/{}'.format(i+1, n_data_files))

    print("\tCreating IMU Map...")
    IMU_aclm_map.append(get_IMU_map(IMU_aclm_fltd[i], data_file_names[i], Fs_sensor, IMU_aclm_threshold, IMU_dilation_time, data_format))
    # IMU_map[i] = np.arange(0, len(IMU_map[i]), 1)
    #Make IMU map 'False' for this dummy data, because we do not know exact threshold
    #IMU_map = [~arr for arr in IMU_map]
    
    # Creating IMU_rotation map
    IMU_RPY_map.append(get_IMU_rot_map(IMU_rotation_fltd[i], IMU_rot_threshold, IMU_dilation_time, Fs_sensor))
    
    IMU_map.append(get_merged_map(IMU_aclm_map[i], IMU_RPY_map[i]))      
    # IMU_map[i][:] = False # New change   
    # IMU_map[i] = IMU_aclm_map[i] # New change    
    
    # ----------------------- Creation of M_sensation_map ----------------%
    # get_sensation_map() function is used here. This function gives the
    # sensation map by dilating every detection to past and future. It also
    # revomes the windows that overlaps with IMU_map. Settings for the
    # extension are given inside the function.
    
    print("\tCreating Matarnal Sensation Map...")
    M_sntn_map.append( get_sensation_map(sensation_data_trimd[i], IMU_map[i],ext_backward, ext_forward, Fs_sensor, Fs_sensation) )
    
    # -----------------------Calculating the number of maternal sensation detection------
    sensation_data_labeled       = label(sensation_data_trimd[i])
    n_Maternal_detected_movement_raw += len(np.unique(sensation_data_labeled)) - 1 #one label is 0 for all 0s, 1 is deducted to remove the initial value of 0
    
    
    sensation_data_labeled_       = label(M_sntn_map[i])
    n_Maternal_detected_movement_after += len(np.unique(sensation_data_labeled_)) - 1 #one label is 0 for all 0s, 1 is deducted to remove the initial value of 0
    
    
    

    print("\tCreating Segmentation of FM data...")
    # ---------------------- Segmentation of FM data ---------------------%
    # Here we will threshold the data, remove body movement, and dilate the data.
    # Setting for the threshold and dilation are given in the function.
    # Finally sensor data will be combined using different Fusion Scheme
    
    sensor_data_fltd = [s1_fltd[i], s2_fltd[i], s3_fltd[i], s4_fltd[i], s5_fltd[i], s6_fltd[i]]
    
    #Get segmented map for each sensors
    sensor_data_sgmntd, threshold[i, 0:n_FM_sensors] = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map[i], FM_dilation_time, Fs_sensor)
    
    #Combine sensor segmentation
    sensor_data_sgmntd_cmbd_all_sensors = sensor_data_sgmntd[0] | sensor_data_sgmntd[1] | sensor_data_sgmntd[2] | sensor_data_sgmntd[3] | sensor_data_sgmntd[4] | sensor_data_sgmntd[5]
  

    # label all detections with chronological numbers starting from 0
    sensor_data_sgmntd_cmbd_all_sensors_labeled = label(sensor_data_sgmntd_cmbd_all_sensors)
    # Number of labels in the sensor_data_cmbd_all_sensors_labeled
    n_label = len(np.unique(sensor_data_sgmntd_cmbd_all_sensors_labeled)) - 1
    
    #label() function returs 2D array, we need 1D array for calculaiton.
    # sensor_data_sgmntd_cmbd_all_sensors_labeled = sensor_data_sgmntd_cmbd_all_sensors_labeled.reshape((sensor_data_sgmntd_cmbd_all_sensors_labeled.size,)) #(1,n) to (n,) array
    

    # ==============================================================================
    # ============================= SENSOR FUSION ==================================
    # ==============================================================================
    
    # SCHEME TYPE 1: ATLEAST n OF the SENSOR ***************************************
    
    # %   All the sensor data are first combined with logical OR. Each
    # %   non-zero sengment in the combined data is then checked against
    # %   individual sensor data to find its presence in that data set.
    # %   Combined data are stored as a cell to make it compatable with the
    # %   function related to matcing with maternal sensation.
    
    #Initialize segmentation map with 
    sensor_data_sgmntd_atleast_1_sensor = np.zeros_like(sensor_data_sgmntd[0])
    sensor_data_sgmntd_atleast_2_sensor = np.zeros_like(sensor_data_sgmntd[0])
    sensor_data_sgmntd_atleast_3_sensor = np.zeros_like(sensor_data_sgmntd[0])
    sensor_data_sgmntd_atleast_4_sensor = np.zeros_like(sensor_data_sgmntd[0])
    sensor_data_sgmntd_atleast_5_sensor = np.zeros_like(sensor_data_sgmntd[0])
    sensor_data_sgmntd_atleast_6_sensor = np.zeros_like(sensor_data_sgmntd[0])

    
    
    if n_label > 0:
        for index in range(1, n_label + 1):
            L_min = np.where(sensor_data_sgmntd_cmbd_all_sensors_labeled == index)[0][0]  # Start of the label
            L_max = np.where(sensor_data_sgmntd_cmbd_all_sensors_labeled == index)[0][-1] + 1  # End of the label

            indv_detection_map = np.zeros_like(sensor_data_sgmntd[0])
            indv_detection_map[L_min:L_max] = 1  # Map individual detection data
            tmp_var = sum([np.any(indv_detection_map * each_sensor_data_sgmntd) for each_sensor_data_sgmntd in sensor_data_sgmntd])
                            
            # For detection by at least n type of sensors
            if tmp_var >= 1:
                sensor_data_sgmntd_atleast_1_sensor[L_min:L_max] = 1
            if tmp_var >= 2:
                sensor_data_sgmntd_atleast_2_sensor[L_min:L_max] = 1
            if tmp_var >= 3:
                sensor_data_sgmntd_atleast_3_sensor[L_min:L_max] = 1
            if tmp_var >= 4:
                sensor_data_sgmntd_atleast_4_sensor[L_min:L_max] = 1
            if tmp_var >= 5:
                sensor_data_sgmntd_atleast_5_sensor[L_min:L_max] = 1
            if tmp_var >= 6:
                sensor_data_sgmntd_atleast_6_sensor[L_min:L_max] = 1
                
                
    # SCHEME TYPE 2: ATLEAST n TYPE OF SENSOR after Combining left and right sensors of each type*****************
    
    # %   First each type of left and right sensor pairs are combined with logical OR.
    # %   Then each non-zero sengment of 'sensor_data_sgmntd_cmbd_all_sensors' is 
    # %   checked against segmentated map of combination of left and right sensor
    # %   pair of each kind to find its presence in that data set.
    
    if data_format == "1":
        #For Old belt
        sensor_data_sgmntd_Left_OR_Right_Aclm  = [sensor_data_sgmntd[0] | sensor_data_sgmntd[1]]
        sensor_data_sgmntd_Left_OR_Right_Acstc = [sensor_data_sgmntd[2] | sensor_data_sgmntd[3]]
        sensor_data_sgmntd_Left_OR_Right_Pzplt = [sensor_data_sgmntd[4] | sensor_data_sgmntd[5]]
        
        #Keep merged sensor data in a list
        sensor_data_sgmntd_cmbd_multi_type_sensors_array = [sensor_data_sgmntd_Left_OR_Right_Aclm, sensor_data_sgmntd_Left_OR_Right_Acstc, sensor_data_sgmntd_Left_OR_Right_Pzplt]
        
    elif data_format == "2":
        #For New belt
        sensor_data_sgmntd_Left_OR_Right_Aclm        = [sensor_data_sgmntd[0] | sensor_data_sgmntd[1]]
        sensor_data_sgmntd_Left_OR_Right_Pzplt_large = [sensor_data_sgmntd[2] | sensor_data_sgmntd[5]]
        sensor_data_sgmntd_Left_OR_Right_Pzplt_small = [sensor_data_sgmntd[3] | sensor_data_sgmntd[4]]
        
        #Keep merged sensor data in a list
        sensor_data_sgmntd_cmbd_multi_type_sensors_array = [sensor_data_sgmntd_Left_OR_Right_Aclm, sensor_data_sgmntd_Left_OR_Right_Pzplt_large, sensor_data_sgmntd_Left_OR_Right_Pzplt_small]
    
    # sensor_data_sgmntd_cmbd_all_sensors_labeled = sensor_data_sgmntd_cmbd_all_sensors_labeled.reshape((sensor_data_sgmntd_cmbd_all_sensors_labeled.size,))  # (1,n) to (n,) array
    
    
    sensor_data_sgmntd_atleast_1_type_of_sensor = np.zeros_like(sensor_data_sgmntd[0])
    sensor_data_sgmntd_atleast_2_type_of_sensor = np.zeros_like( sensor_data_sgmntd[0])
    sensor_data_sgmntd_atleast_3_type_of_sensor = np.zeros_like( sensor_data_sgmntd[0])

    if n_label > 0:
        for k in range(1, n_label + 1):
            L_min = np.where(sensor_data_sgmntd_cmbd_all_sensors_labeled == k)[0][0]  # Start of the label
            L_max = np.where(sensor_data_sgmntd_cmbd_all_sensors_labeled == k)[0][-1] + 1  # End of the label

            indv_detection_map = np.zeros_like(sensor_data_sgmntd[0])
            # Map individual detection data
            indv_detection_map[L_min:L_max] = 1
            tmp_var = sum([np.any(indv_detection_map * temp_sensor_data_sgmntd)
                          for temp_sensor_data_sgmntd in sensor_data_sgmntd_cmbd_multi_type_sensors_array])
            if tmp_var >= 1:
                sensor_data_sgmntd_atleast_1_type_of_sensor[L_min:L_max] = 1
            if tmp_var >= 2:
                sensor_data_sgmntd_atleast_2_type_of_sensor[L_min:L_max] = 1
            if tmp_var >= 3:
                sensor_data_sgmntd_atleast_3_type_of_sensor[L_min:L_max] = 1


    

    schemes = [sensor_data_sgmntd_atleast_1_type_of_sensor, sensor_data_sgmntd_atleast_2_type_of_sensor, sensor_data_sgmntd_atleast_3_type_of_sensor, sensor_data_sgmntd_atleast_1_sensor,
               sensor_data_sgmntd_atleast_2_sensor, sensor_data_sgmntd_atleast_3_sensor, sensor_data_sgmntd_atleast_4_sensor, sensor_data_sgmntd_atleast_5_sensor, sensor_data_sgmntd_atleast_6_sensor]
    
    all_data_files_schemes.append(schemes)
    
# =============================================================================
#     # Fix the scheme based on which user wants to extract TPD and FPD
# =============================================================================
    desired_scheme = 2
    user_scheme = schemes[desired_scheme]
    # ---------------------- Extraction of TPDs and FPDs -----------------------
    print("\tExtracting TPDs and FPDs Based on Schemes")
    
    user_scheme_labeled = label(np.array(user_scheme)) # taking as numpy array since lable function expects an array
    n_label = len(np.unique(user_scheme_labeled)) - 1 # Number of labels in the sensor_data_cmbd_all_sensors_labeled
    
    #label() function returs 2D array, we need 1D array for calculaiton.
    user_scheme_labeled = user_scheme_labeled.reshape((user_scheme_labeled.size,)) #(1,n) to (n,) array
 
    if n_label: # When there is a detection by the sensor system
        TPD_extracted_single, FPD_extracted_single, extracted_TPD_weightage_single,\
        extracted_FPD_weightage_single = extract_detections(M_sntn_map[i],
                                                            sensor_data_fltd,
                                                            sensor_data_sgmntd,
                                                            user_scheme_labeled,
                                                            n_label, n_FM_sensors)
        
        #append in list variable
        TPD_extracted.append(TPD_extracted_single)
        FPD_extracted.append(FPD_extracted_single)
        extracted_TPD_weightage.append(extracted_TPD_weightage_single)
        extracted_FPD_weightage.append(extracted_FPD_weightage_single)
    
    '''
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data extraction ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("\tExtracting TPDs and FPDs")
    for scheme_ind, scheme in enumerate(schemes):

        # taking as numpy array since lable function expects an array
        scheme_labeled = label(scheme)
        scheme_labeled_reshaped = scheme_labeled.reshape( (scheme_labeled.size,))
        
        # Number of labels in the sensor_data_cmbd_all_sensors_labeled
        n_label_scheme = len(np.unique(label(scheme_labeled_reshaped))) - 1

        # if n_label_scheme:  # When there is a detection by the sensor system
        #     TPD_extracted_single, FPD_extracted_single, extracted_TPD_weightage_single,\
        #         extracted_FPD_weightage_single = extract_detections(M_sntn_map[i],
        #                                                             sensor_data_fltd,
        #                                                             sensor_data_sgmntd,
        #                                                             scheme_labeled_reshaped,
        #                                                             n_label_scheme, n_FM_sensors)

        #     # append in list variable
        #     TPD_extracted.append(TPD_extracted_single)
        #     FPD_extracted.append(FPD_extracted_single)
        #     extracted_TPD_weightage.append(extracted_TPD_weightage_single)
        #     extracted_FPD_weightage.append(extracted_FPD_weightage_single)

        # copy creates an independent array
        current_ML_detection_map = np.copy([scheme])

        TPD_indv[i, scheme_ind], FPD_indv[i, scheme_ind], TND_indv[i, scheme_ind], FND_indv[i, scheme_ind] = match_with_m_sensation(
            current_ML_detection_map, sensation_data_trimd[i], IMU_map[i], M_sntn_map[i], ext_backward,
            ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation)
    '''
    
# Clearing unnecessary variables
#del sensor_data_fltd, sensor_data_sgmntd, sensor_data_sgmntd_cmbd_all_sensors, sensor_data_sgmntd_cmbd_all_sensors_labeled

# Ending notification
print('\nData extraction is completed.\n')

# ~~~~~~~~~~~~~~~~~~~~~~~~~ Feature extraction ~~~~~~~~~~~~~~~~~~~~~~~~~~~%
# In this part of the code, selected features are extracted from the
# the TPD and FPD data sets extracted in the previous step.


print('\n#Feature extraction for machine learning is going on\n...')

X_TPD, X_FPD, n_TPD, n_FPD, total_duration_TPD, total_duration_FPD = extract_features(
    TPD_extracted, extracted_TPD_weightage, FPD_extracted, extracted_FPD_weightage,
    threshold, Fs_sensor, n_FM_sensors)

print('\nFeature extraction is completed.')
print('\nIn total, %d features were collected from all the sensor data.\n' % X_TPD.shape[1])

# Combining features from the TPD and FPD data sets
X = np.vstack((X_TPD, X_FPD))
Y = np.zeros((n_TPD + n_FPD, 1))
Y[:n_TPD, 0] = 1

Y = np.ravel(Y) # To change shape from (n,1) to (n,) 1D array

# Feature normalization
X_norm, _, _ = normalize_features(X)

# As all the rows captured by getMPDATOM and getTFMF gives 1 and 0 respectively, the normalized values give nan 
# due to the zero division error. So, dropping all the columns with nan values
nan_columns = np.any(np.isnan(X_norm), axis=0)
X_norm = X_norm[:, ~nan_columns]


X_TPD_norm = X_norm[:X_TPD.shape[0], :]  # Normalize with respect to deviation (= max - min)
X_FPD_norm = X_norm[X_TPD.shape[0]:, :]  # Normalize with respect to deviation (= max - min)


print("Time taken: ", time.time()- tic)

#%% FEATURE RANKING 

"""
@author: Monaf Chowdhury
Provides suitable option for feature selection in order to reduce feature space. Two options. 
ONE: Using old matlab features selected using NCA
TWO: Ensemble feature selection using four SOTA feature selection methods. i.e,  NCA, XGBoost, L1 based, Recursive feature selection 
"""

feature_ranker = FeatureRanker(X_norm, Y) # Creates an instance to the class FeatureRanker
n = 100 # Ideally the number of features that we want 
print("/n# Enter the Feature selection method ")
print("1 - Old MATLAB features")
print("2 - Ensemble Method using four feature selection methodologies")
# feature_selection_method = int(input("Feature Selection Method: "))
feature_selection_method = 2 
if feature_selection_method == 1: 
    matlab_top_n = feature_ranker.use_matlab_nca_features() # Old features extracted from matlab
    index_top_features = matlab_top_n
else:
    # fusion_criteria = int(input("\nHow many feature selection methods must have common features: "))
    fusion_criteria = 2 
    # Identifying top features from NCA, XGBoost, L1 based feature selection and Recursive feature selection method
    nca_top_n = feature_ranker.nca_ranking(n)   # Features selected using Neighbourhood component analysis 
    xgb_top_n = feature_ranker.xgboost_ranking(n) # Features selected using XGBoost 
    l1_based_top_n = feature_ranker.logistic_regression_ranking(n) # Features selected using L1 based feature selection
    recursive_top_n = feature_ranker.recursive_feature_elimination(n) # Features selected using Recursive elimination method
    
    # Finding out the features using ensemble feature selection method
    index_top_features = feature_ranker.ensemble_feature_selection(fusion_criteria, nca_top_n, xgb_top_n, l1_based_top_n, recursive_top_n) # pass the feature selection methods

print(f"Total number of features: {len(index_top_features)}")
print(index_top_features)


X_TPD_norm_ranked = X_TPD_norm[:, index_top_features]
X_FPD_norm_ranked = X_FPD_norm[:, index_top_features]

# Saving of non-randomized features in text files
np.savetxt('X_TPD_norm_ranked.txt', X_TPD_norm_ranked)
np.savetxt('X_FPD_norm_ranked.txt', X_FPD_norm_ranked)

# X_TPD_norm_ranked = np.genfromtxt('D:/Monaf/From GITHUB/FMdata-analysis/X_TPD_norm_ranked.txt')
# X_FPD_norm_ranked = np.genfromtxt('D:/Monaf/From GITHUB/FMdata-analysis/X_FPD_norm_ranked.txt')

#%% PREPARATION OF TRAINING AND TESTING DATA FILES =========================

 # Division into test and training sets
 #   1: Divide by hold out method 
 #   2: K-fold with original ratio of FPD and TPD in each fold, 
 #   3: K-fold with custom ratio of FPD and TPD in each fold. 
 #   4: Divide by participants
 #   In the cases of option 1, 2 and 3, stratified division will be created, 
 #   i.e. each division will have the same ratio of FPD and TPD.



data_div_option = 2  # Change this value based on the data division option you want

if data_div_option == 1:  # Divide data using the holdout method
    training_portion = 0.8  # Training portion in case of the holdout method- option 1
    X_train,Y_train,X_test,Y_test,n_training_data_TPD,n_training_data_FPD,n_test_data_TPD,\
        n_test_data_FPD = divide_by_holdout(X_TPD_norm_ranked,X_FPD_norm_ranked,training_portion)
    
    print('Data division is completed.')
    print(f'Data was divided using stratified holdout method with {training_portion*100:.2f}% - {(1-training_portion)*100:.2f}% ratio.')

elif data_div_option in [2, 3]:  # Divide data using stratified K-folds
    K = 5  # Number of folds
    X_K_fold, Y_K_fold, n_data_TPD_each_fold, n_data_TPD_last_fold, n_data_FPD_each_fold,\
    n_data_FPD_last_fold,FPD_TPD_ratio,rand_num_TPD,rand_num_FPD\
        = divide_by_K_folds(X_TPD_norm_ranked,X_FPD_norm_ranked,data_div_option,K)
    
    print('Data division is completed.')
    print(f'Data was divided into stratified {K}-folds with FPD/TPD ratio of {np.mean(FPD_TPD_ratio):.2f}')

elif data_div_option == 4:  # Divide data by participant
    X_TPD_by_participant, X_FPD_by_participant \
        = divide_by_participants(data_file_names, TPD_extracted, FPD_extracted, 
                                 X_TPD_norm_ranked, X_FPD_norm_ranked)
        
    print('Data division by participants is not implemented yet.')

else:
    print('Wrong data division option chosen.')


#%% CROSS-VALIDATION TO FIND THE OPTIMUM HYPERPARAMETERS ====================
# ------------- Cross-validation to select model parameters ---------------

print('\nSelecting the model parameters through cross-validation ...')
tic = time.time()

# Variable declaration and definition
# n_participant = len(T_general_info["Participant ID"].unique())  # Number of participants in the data set
n_participant = 5   # Coomenting the previous line as we don't use any excel files

if data_div_option == 1:
    n_iter = 1  # Iteration to loop across different participants/folds
elif data_div_option in [2, 3]:
    n_iter = K
elif data_div_option == 4:
    n_iter = n_participant


train_Accuracy = np.zeros(n_iter)
test_Accuracy = np.zeros(n_iter)
test_Accuracy_TPD = np.zeros(n_iter)
test_Accuracy_FPD = np.zeros(n_iter)

cost_TPD      = FPD_TPD_ratio # Higher waitage for TPD 
# cost_TPD = 2  # Higher weightage for getting TPD wrong
cost_function = {0:1, 1:cost_TPD}


# Classifier models
SVM_model = list(np.zeros(n_iter,dtype=int)) #Contains SVM models
NN_model = list(np.zeros(n_iter,dtype=int)) #Contains NN models
LR_model = list(np.zeros(n_iter,dtype=int)) #Contains LR models
RF_model = list(np.zeros(n_iter,dtype=int)) #Contains RF models

# Options for classification
classifier_option = 3  # 1-LR, 2-SVM, 3-NN, 4-RF. Only the selected classifier will be used
x_validate_KxK_option = 1  # 1: cross-validate by KxK-fold; 0: cross-validate by K-fold only


# Neural Network Variable decleration and initialization
min_unit_per_layer = 10
max_unit_per_layer = 200
step_size = 10
n_iter_NN = (max_unit_per_layer-min_unit_per_layer)//step_size+1
history = []

Y_val_prediction_TPD_rand = np.zeros((1, 1))
Y_val_prediction_FPD_rand = np.zeros((1, 1))
train_accuracy = np.zeros((K, n_iter_NN))
val_accuracy = np.zeros((K, n_iter_NN))

for i in range(n_iter):
    print(f'\nCurrent iteration: {i + 1}/{n_iter}')

    # ------------ Selection of training and testing data sets ------------
    if data_div_option == 1:  # Based on holdout method
        X_train_current = X_train
        Y_train_current = Y_train

        X_test_current = X_test
        Y_test_current = Y_test

        n_test_data_TPD_current = n_test_data_TPD
        n_test_data_FPD_current = n_test_data_FPD

    elif data_div_option in [2, 3]:  # Based on stratified K-fold division
        # Partitioning data for K x K nested cross-validation
        # Assign i-th fold as the test data
        X_test_current = X_K_fold[i]
        Y_test_current = Y_K_fold[i]
    
        # Assign rest of the folds as the training data
        X_train_current = np.zeros( (1, X_K_fold[i].shape[1]) )
        Y_train_current = np.zeros((1))
    
        for j in range(K):
            if j != i:
                X_train_current = np.vstack((X_train_current, X_K_fold[j]))
                Y_train_current = np.append(Y_train_current, Y_K_fold[j])
        
        #remove first invalid row while initialized
        X_train_current = X_train_current[1:]
        Y_train_current = Y_train_current[1:]
        
        
        if i == n_iter-1:
            n_test_data_TPD_current = n_data_TPD_last_fold
            n_test_data_FPD_current = n_data_FPD_last_fold
        else:
            n_test_data_TPD_current = n_data_TPD_each_fold
            n_test_data_FPD_current = n_data_FPD_each_fold 
    
        # Converting Y_train_current to a 1D array if needed
        Y_train_current = Y_train_current.ravel()
    
    elif data_div_option == 4:  # Based on data division by participants
        X_test_current = np.vstack((X_TPD_by_participant[i], X_FPD_by_participant[i]))
        Y_test_current = np.hstack((np.ones(len(X_TPD_by_participant[i])), np.zeros(len(X_FPD_by_participant[i]))))

        X_train_current, Y_train_current = None, None
        for j in range(n_participant):
            if j != i:
                X_train_current = np.vstack((X_train_current, X_TPD_by_participant[j], X_FPD_by_participant[j]))
                Y_train_current = np.hstack((Y_train_current, np.ones(len(X_TPD_by_participant[j])),
                                              np.zeros(len(X_FPD_by_participant[j]))))

        n_test_data_TPD_current, n_test_data_FPD_current = len(X_TPD_by_participant[i]), len(X_FPD_by_participant[i])
        X_train_current = X_train_current[1:]  # Removing the 1st row with zero values
        Y_train_current = Y_train_current[1:]

    else:
        print('\nWrong data division option chosen.')
        break

    # ------------------ Cross-validating the model ---------------------
    if classifier_option == 1:  # Logistic regression
        # Training the model
        LR_model[i] = LogisticRegressionCV(Cs=20, penalty='l2', cv=5, class_weight = cost_function,  solver='lbfgs', max_iter=1000, verbose=False, n_jobs=-1)
        #Cs (int  or list of floats)-
            #If Cs is as an int, then a grid of Cs values are chosen in a logarithmic scale between 1e-4 and 1e4.
            #If Cs is list of floats, each of the values in Cs describes the inverse of regularization strength.
            #Like in support vector machines, smaller values specify stronger regularization.
        
        #cv - No of folds for cross validation
        
        #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        
        LR_model[i].fit(X_train_current, Y_train_current)
                
        # Getting prediction accuracies using the function
        train_Accuracy[i], test_Accuracy[i], test_Accuracy_TPD[i], test_Accuracy_FPD[i]\
            = get_prediction_accuracies(LR_model[i], X_train_current, Y_train_current,
                                        X_test_current, Y_test_current, 
                                        n_test_data_TPD_current, n_test_data_FPD_current)
    
    
    elif classifier_option == 2:  # SVM
        # Training the model
        # SVM_model[i] = SVC(kernel='rbf', class_weight = cost_function, verbose=True)
        # SVM_model[i].fit(X_train_current, Y_train_current)
        
        # Create the SVM classifier with specific options
        t = SVC(kernel='rbf', class_weight=cost_function, random_state=0, verbose=False)
        
        # Define the hyperparameter grid for the optimization
        param_grid = {
            #Regularization strength chosen in logarithmic scale between 1e-4 and 1e4
            'C': [0.0001,0.000263665,0.000695193,0.00183298,0.00483293,0.0127427,0.0335982,0.0885867,0.233572,0.615848,1.62378,4.28133,11.2884,29.7635,78.476,206.914,545.559,1438.45,3792.69,10000],
            'gamma': [0.0001,0.000263665,0.000695193,0.00183298,0.00483293,0.0127427,0.0335982,0.0885867,0.233572,0.615848,1.62378,4.28133,11.2884,29.7635,78.476,206.914,545.559,1438.45,3792.69,10000]
        }
        #BoxConstraint(C), and KernelScale(gamma) are determined based on a K-fold cross-validation
        
        # Create the Repeated Stratified K-Fold cross-validation object (assuming 5-fold cross-validation repeated 3 times)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
        
        # GridSearchCV for hyperparameter optimization
        grid_search = GridSearchCV(
            estimator=t,
            param_grid=param_grid,
            scoring='accuracy',  # Replace 'accuracy' with appropriate scoring metric
            cv=cv,
            n_jobs=-1,  # Use -1 for utilizing all available CPU cores
            verbose=2
        )
        
        # Fit the model to the data and perform hyperparameter optimization
        grid_search.fit(X_train_current, Y_train_current)
        
        # Get the best model
        SVM_model[i] = grid_search.best_estimator_

        train_Accuracy[i], test_Accuracy[i], test_Accuracy_TPD[i], test_Accuracy_FPD[i]\
            = get_prediction_accuracies(SVM_model[i], X_train_current, Y_train_current,
                                        X_test_current, Y_test_current, 
                                        n_test_data_TPD_current, n_test_data_FPD_current)
    

    elif classifier_option == 3:  # NN
        # Training the model
        print('\nCross-validating the Neural Network algorithm...\n')    
        
        #Test data is considered as validation set for Neural Network
        X_val_current = X_test_current
        Y_val_current = Y_test_current

        # Iterate across network architectures and find training and validation accuracy
        for j in range(n_iter_NN):

            # Define the model
            # Gives the number of features as the input layer size
            n_input = X_train_current.shape[1]
            n_hidden_layer = 1
            n_unit_per_layer = min_unit_per_layer+j*step_size

            # This is a user defined function that returns the model
            model = define_model(n_input, n_hidden_layer, n_unit_per_layer)
            print('\t\tHiddend layer size: ', n_unit_per_layer)

            # Train the model
            # 2.71 is the ratio of FPD and TPD in case of all sensors used for feature extraction
            weight_TPD = 2
            weight_FPD = 1
            # this means the weitage of class 0 is 1 and class 1 is 2
            overall_weights = {0: 1, weight_FPD: weight_TPD}
            training_history = model.fit(X_train_current, Y_train_current, class_weight=overall_weights,
                                         epochs=100, shuffle=False, verbose=0)  # verbose = 0 stops displaying calcuation after each epoch
            history.append(training_history) # line added by monaf to check model's training history
            # Evaluate the model
            threshold_NN = 0.5  # threshold value for prediction
            
            #------------Calculating Taining Accuracy-------------
            Y_hat_train = model.predict(X_train_current)
            Y_hat_train = expit(Y_hat_train) # Converts to probability value by applying sigmoid function
            Y_train_prediction = (Y_hat_train >= threshold_NN).astype(int)  # thresholding and converting to integer from bool
            train_accuracy[i, j] = accuracy_score(Y_train_current, Y_train_prediction)

            #------------Calculating Validation Accuracy----------
            Y_hat_val = model.predict(X_val_current)
            Y_hat_val = expit(Y_hat_val) # Converts to probability value by applying sigmoid function
            Y_val_prediction = (Y_hat_val >= threshold_NN).astype(int) # thresholding and converting to integer from bool
            val_accuracy[i, j] = accuracy_score(Y_val_current, Y_val_prediction)
            

            # Print the performance in each iteration
            print('\t Train accuracy: \t\t %.3f' %train_accuracy[i,j])
            print('\t Validation accuracy : \t %.3f\n' %val_accuracy[i,j])

        # print('Validation of the neural network is completed.')
        train_Accuracy[i]= np.average(train_accuracy[i,:])
        test_Accuracy[i] = np.average(val_accuracy[i,:])
        

    elif classifier_option == 4:  # Random forest
        # Training the model
        # RF_model[i] = RandomForestClassifier(n_estimators=100, min_samples_leaf=50, random_state=0)
        # RF_model[i].fit(X_train_current, Y_train_current)
        
        # Create the random forest classifier with specific options
        t = RandomForestClassifier(n_estimators=100, min_samples_leaf=50, class_weight=cost_function, n_jobs=-1, random_state=0)
        
        # Define the hyperparameter grid for the optimization
        param_grid = {
            'max_features': ['sqrt', 'log2', None, 1,2,3,4,5,6,8,10,12,14,16,17,19,22,25,27,29]
            # - If "sqrt", then `max_features=sqrt(n_features)`.
            # - If "log2", then `max_features=log2(n_features)`.
            # - If None, then `max_features=n_features`.
        }
        # Random forest algorithm is used for tree ensemble.
        # Minimum size of the leaf node is used as the stopping criterion.
        # Number of features to sample in each tree are determined based on a K-fold cross-validation.
        
        # Create the Repeated Stratified K-Fold cross-validation object (assuming 5-fold cross-validation repeated 2 times)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
        
        # GridSearchCV for hyperparameter optimization
        grid_search = GridSearchCV(
            estimator=t,
            param_grid=param_grid,
            scoring='accuracy',  # Replace 'accuracy' with appropriate scoring metric
            cv=cv,
            n_jobs=-1,  # Use -1 for utilizing all available CPU cores
            verbose=2
        )
        
        # Fit the model to the data and perform hyperparameter optimization
        grid_search.fit(X_train_current, Y_train_current)
        
        # Get the best model
        RF_model[i] = grid_search.best_estimator_


        train_Accuracy[i], test_Accuracy[i], test_Accuracy_TPD[i], test_Accuracy_FPD[i]\
            = get_prediction_accuracies(RF_model[i], X_train_current, Y_train_current,
                                        X_test_current, Y_test_current, 
                                        n_test_data_TPD_current, n_test_data_FPD_current)
    


    else:
        print('\nWrong classifier option chosen.')
        break

    if x_validate_KxK_option == 0:
        break  # Break after 1st loop.

print("\nBest Accuracy for each fold")
print("=============================")
print("train_Accuracy\n", train_Accuracy)
print("test_Accuracy\n", test_Accuracy)
print("test_Accuracy_TPD\n", test_Accuracy_TPD)
print("test_Accuracy_FPD\n", test_Accuracy_FPD)

# Find the best model
I_max = np.argmax(test_Accuracy)  # Index of the model with the maximum test accuracy

print('Model parameter selection is done.')
print("Time taken: ", time.time()- tic)

#%% TRAINING AND TESTING THE MODEL WITH OPTIMUM HYPERPARAMTERS =============   

print("\nTraining the selected model for K-fold testing errors")
tic = time.time()

# Variable declaration and definition
# n_participant = len(T_general_info["Participant ID"].unique())  # Number of participants in the data set

if data_div_option == 1:
    n_iter = 1  # Iteration to loop across different participants/folds
elif data_div_option in [2, 3]:
    n_iter = K
elif data_div_option == 4:
    n_iter = n_participant

final_train_Accuracy    = np.zeros(n_iter)
final_test_Accuracy     = np.zeros(n_iter)
final_test_Accuracy_TPD = np.zeros(n_iter)
final_test_Accuracy_FPD = np.zeros(n_iter)
final_test_prediction   = list(np.zeros(n_iter, dtype=int))
final_test_scores       = list(np.zeros(n_iter, dtype=int))


# Variables for the selected models
if classifier_option == 1:  # Logistic regression
    LR_model_selected = list(np.zeros(n_iter,dtype=int))
elif classifier_option == 2:  # SVM
    SVM_model_selected = list(np.zeros(n_iter,dtype=int))
elif classifier_option == 3:  # NN
    NN_model_selected = list(np.zeros(n_iter,dtype=int))
elif classifier_option == 4:  # Random forest
    RF_model_selected = list(np.zeros(n_iter,dtype=int))


#Variable for Neural Network
Y_test_prediction_TPD_rand = np.zeros((1, 1))
Y_test_prediction_FPD_rand = np.zeros((1, 1))
history = []

# Because the model parameters are fixed now, we can find the K-fold test
# error for the same model parameter and get the average and SD.
for i in range(n_iter):
    print(f'\nCurrent iteration: {i + 1}/{n_iter}')

    # ------------ Selection of training and testing data sets ------------
    if data_div_option == 1:  # Based on holdout method
        X_train_current = X_train
        Y_train_current = Y_train

        X_test_current = X_test
        Y_test_current = Y_test

        n_test_data_TPD_current = n_test_data_TPD
        n_test_data_FPD_current = n_test_data_FPD

    elif data_div_option in [2, 3]:  # Based on stratified K-fold division
        # Partitioning data for K x K nested cross-validation
        #   Assign i-th fold as the test data
        X_test_current = X_K_fold[i]
        Y_test_current = Y_K_fold[i]

        # Assign rest of the folds as the training data
        X_train_current = np.zeros( (1, X_K_fold[i].shape[1]) )
        Y_train_current = np.zeros((1))
        
        
        for j in range(K):
            if j != i:
                X_train_current = np.vstack((X_train_current, X_K_fold[j]))
                Y_train_current = np.append(Y_train_current, Y_K_fold[j])
        
        #remove first invalid row while initialized
        X_train_current = X_train_current[1:]
        Y_train_current = Y_train_current[1:]

        if i == n_iter - 1:
            n_test_data_TPD_current = n_data_TPD_last_fold
            n_test_data_FPD_current = n_data_FPD_last_fold
        else:
            n_test_data_TPD_current = n_data_TPD_each_fold
            n_test_data_FPD_current = n_data_FPD_each_fold

    elif data_div_option == 4:  # Based on data division by participants
        X_test_current = np.vstack((X_TPD_by_participant[i], X_FPD_by_participant[i]))
        Y_test_current = np.zeros((X_test_current.shape[0], 1))
        Y_test_current[:X_TPD_by_participant[i].shape[0]] = 1

        X_train_current = np.zeros((1, X_test_current.shape[1]))
        Y_train_current = np.zeros((1, 1))
        for j in range(n_participant):
            if j != i:
                X_train_current = np.vstack((X_train_current, X_TPD_by_participant[j], X_FPD_by_participant[j]))
                Y_train_current = np.vstack((Y_train_current, np.ones((X_TPD_by_participant[j].shape[0], 1)), np.zeros((X_FPD_by_participant[j].shape[0], 1))))
        X_train_current = X_train_current[1:, :]  # Removing the 1st row with zero values
        Y_train_current = Y_train_current[1:, :]

        n_test_data_TPD_current = X_TPD_by_participant[i].shape[0]
        n_test_data_FPD_current = X_FPD_by_participant[i].shape[0]
    else:
        print('\nWrong data division option chosen. \n')
        break

    
    # ----------------------- Training the model --------------------------
    if classifier_option == 1:  # Logistic regression
        # Training the model
        C_selected = LR_model[I_max].C_[0]  # lambda with lowest test error
        LR_model_selected[i] = LogisticRegression(C=C_selected, class_weight = cost_function, solver='lbfgs',max_iter=1000, verbose=False, n_jobs=-1)
        
        LR_model_selected[i].fit(X_train_current, Y_train_current)
        
        final_train_Accuracy[i], final_test_Accuracy[i], final_test_Accuracy_TPD[i], final_test_Accuracy_FPD[i]\
            = get_prediction_accuracies(LR_model_selected[i], X_train_current, Y_train_current,
                                        X_test_current, Y_test_current, 
                                        n_test_data_TPD_current, n_test_data_FPD_current)
        
        final_test_scores[i]        = LR_model_selected[i].predict_proba(X_test_current)
        final_test_prediction[i]    = LR_model_selected[i].predict(X_test_current)


    elif classifier_option == 2:  # SVM
        # Training the model
        C = SVM_model[I_max].C  # Box constraint
        gamma = SVM_model[I_max].gamma  # Kernel scale
        
        SVM_model_selected[i] = SVC(kernel='rbf', C=C, gamma=gamma, class_weight=cost_function, probability=True, random_state=0)
        
        SVM_model_selected[i].fit(X_train_current, Y_train_current)
        
        
        final_train_Accuracy[i], final_test_Accuracy[i], final_test_Accuracy_TPD[i], final_test_Accuracy_FPD[i]\
            = get_prediction_accuracies(SVM_model_selected[i], X_train_current, Y_train_current,
                                        X_test_current, Y_test_current, 
                                        n_test_data_TPD_current, n_test_data_FPD_current)
            
        final_test_scores[i]        = SVM_model_selected[i].predict_proba(X_test_current)
        final_test_prediction[i]    = SVM_model_selected[i].predict(X_test_current)


    elif classifier_option == 3:  # NN
        #FIND THE OPTIMUM NETWORK ARCHITECTURE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Based on the individual max accuracy
        # index_max = np.where(val_accuracy==val_accuracy.max()) # Gives a touple with the location of max value
        # index_row_max = index_max[0][0]
        # index_col_max = index_max[1][0]

        # Based on the overall max accuracy of all folds for a particular architecture
        val_accuracy_overall = val_accuracy.sum(axis=0) #columnwise sum; sum for each Neural Network itereation
        index_col_max = val_accuracy_overall.argmax()

        # Calculate the number of units based on the best validation performance
        n_unit_per_layer_optimum = min_unit_per_layer+index_col_max*step_size
        # n_unit_per_layer_optimum = 140 # 190 from recent optimization.
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx #

        # TEST of THE ALGORITHM TO FIND THE GENERALIZED PERFORMANCE ~~~~~~~~~~~~~~~~~~~
        # print('\nTesting the Neural Network algorithm...\n')
        

        # Define the model
        # Gives the number of features as the input layer size
        n_input = X_train_current.shape[1]
        n_hidden_layer = 1
        n_unit_per_layer = n_unit_per_layer_optimum
        # This is a user defined function that returns the model
        model = define_model(n_input, n_hidden_layer, n_unit_per_layer)

        # Train the model
        weight_TPD = 2
        weight_FPD = 1
        # this means the weightage of class 0 is 1 and class 1 is 2
        overall_weights = {0: 1, weight_FPD: weight_TPD}
        
        # Early stopping to prevent overfitting # callbacks=[early_stopping]
        early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
        
        training_history = model.fit(X_train_current, Y_train_current, class_weight=overall_weights, epochs=100,  validation_split=0.2,
                                     shuffle=False, verbose=0)  # verbose = 0 stops displaying calcuation after each epoch
        history.append(training_history)
        # Evaluate the model
        threshold_NN = 0.5  # threshold value for prediction
        
        #------------Calculating Taining Accuracy-------------
        Y_hat_train = model.predict(X_train_current)
        Y_hat_train = expit(Y_hat_train) # Converts to probability value by applying sigmoid function
        Y_train_prediction = (Y_hat_train >= threshold_NN).astype(int)  # thresholding and converting to integer from bool
        # makes the variable to a column variable
        Y_train_current = Y_train_current[:, np.newaxis]
        train_accuracy = accuracy_score(Y_train_current, Y_train_prediction)
        

        #------------Calculating Test Accuracy----------------
        Y_hat_test = model.predict(X_test_current)
        Y_hat_test = expit(Y_hat_test) # Converts to probability value by applying sigmoid function
        Y_test_prediction = (Y_hat_test >= threshold_NN).astype(int) # Comment this line in case of ROC + PR analysis in Matlab    
        # Y_test_prediction = Y_hat_test    # Uncomment this line in case of ROC+PR analysis in Matlab
        
        # Y_test_current = Y_test_current[:, np.newaxis]# makes the variable to a column variable
        test_accuracy = accuracy_score(Y_test_current, Y_test_prediction)

        if data_div_option == 2 or data_div_option == 3:  # Stratified K-fold division

            if i != (K-1):  # all the folds except the last fold
                test_accuracy_TPD = accuracy_score(Y_test_current[0:n_data_TPD_each_fold], Y_test_prediction[0:n_data_TPD_each_fold])
                test_accuracy_FPD = accuracy_score(Y_test_current[n_data_TPD_each_fold:None], Y_test_prediction[n_data_TPD_each_fold:None])

                Y_test_prediction_TPD_rand = np.concatenate((Y_test_prediction_TPD_rand, Y_test_prediction[0:n_data_TPD_each_fold]))  # Stores the prediction from each cycle
                Y_test_prediction_FPD_rand = np.concatenate((Y_test_prediction_FPD_rand, Y_test_prediction[n_data_TPD_each_fold:None]))  # Stores the prediction from each cycle

            else:  # the last fold
                test_accuracy_TPD = accuracy_score(Y_test_current[0:n_data_TPD_last_fold], Y_test_prediction[0:n_data_TPD_last_fold])
                test_accuracy_FPD = accuracy_score(Y_test_current[n_data_TPD_last_fold:None], Y_test_prediction[n_data_TPD_last_fold:None])

                Y_test_prediction_TPD_rand = np.concatenate((Y_test_prediction_TPD_rand, Y_test_prediction[0:n_data_TPD_last_fold]))  # Stores the prediction from each cycle
                Y_test_prediction_FPD_rand = np.concatenate((Y_test_prediction_FPD_rand, Y_test_prediction[n_data_TPD_last_fold:None]))  # Stores the prediction from each cycle

        # else:
        #     test_accuracy_TPD = np.sum(1*(Y_test_prediction[0:n_TPD_by_P[i]] == Y_test_current[0:n_TPD_by_P[i]]))/n_TPD_by_P[i]
        #     test_accuracy_FPD = np.sum(1*(Y_test_prediction[n_TPD_by_P[i]:None] == Y_test_current[n_TPD_by_P[i]:None]))/n_FPD_by_P[i]

        #     Y_test_prediction_TPD_rand = np.concatenate(
        #         (Y_test_prediction_TPD_rand, Y_test_prediction[0:n_TPD_by_P[i]]))  # Stores the prediction from each cycle
        #     Y_test_prediction_FPD_rand = np.concatenate(
        #         (Y_test_prediction_FPD_rand, Y_test_prediction[n_TPD_by_P[i]:None]))  # Stores the prediction from each cycle

        # Generates ROC-AUC
        ROC_AUC = roc_auc_score(Y_test_current, Y_hat_test)

        # Print the performance in each iteration
        print('\tOverall train accuracy: %.3f' % train_accuracy)
        print('\tOverall test accuracy : %.3f' % test_accuracy)
        print('\tTPD test accuracy     : %.3f' % test_accuracy_TPD)
        print('\tFPD test accuracy     : %.3f' % test_accuracy_FPD)
        print('\tROC AUC               : %.3f\n' % ROC_AUC)

        print('Testing of the neural network is completed.')
        
        final_train_Accuracy[i]     = train_accuracy
        final_test_Accuracy[i]      = test_accuracy
        final_test_Accuracy_TPD[i]  = test_accuracy_TPD
        final_test_Accuracy_FPD[i]  = test_accuracy_FPD
        
    
    elif classifier_option == 4:  # Random forest
        # Training the model
        # RF_NLC = RF_model[I_max].ModelParameters.NLearn  # Number of trees in the ensemble
        t_selected = RF_model[I_max].estimators_[0]
        # NumVariablesToSample to sample and MinLeafSize are included here

        
        RF_model_selected[i] = RandomForestClassifier(n_estimators=100, max_features=t_selected.max_features,
    min_samples_leaf=t_selected.min_samples_leaf, class_weight=cost_function, n_jobs=-1, random_state=0)
        
        RF_model_selected[i].fit(X_train_current, Y_train_current)

        
        final_train_Accuracy[i], final_test_Accuracy[i], final_test_Accuracy_TPD[i], final_test_Accuracy_FPD[i]\
            = get_prediction_accuracies(RF_model_selected[i], X_train_current, Y_train_current,
                                        X_test_current, Y_test_current, 
                                        n_test_data_TPD_current, n_test_data_FPD_current)
        
        #RandomForestClassifier doesn't have a decision_function method like LogisticRegression does.
        #Instead, we can directly use the predict_proba method to get the class probabilities from a RandomForestClassifier.
        final_test_scores[i] = RF_model_selected[i].predict_proba(X_test_current)
        final_test_prediction[i] = RF_model_selected[i].predict(X_test_current)

        

if classifier_option ==3:
    # Generate the cumulative test predctions
    #   Remove the initial row with 0
    Y_test_prediction_TPD_rand = Y_test_prediction_TPD_rand[1:None]
    Y_test_prediction_FPD_rand = Y_test_prediction_FPD_rand[1:None]

    Y_test_prediction_TPD = np.zeros((n_TPD, 1))
    Y_test_prediction_FPD = np.zeros((n_FPD, 1))

    if data_div_option == 2 or data_div_option == 3:  # Stratified K-fold division
        #   Non-randomized the predictions to match with the original data set
        for i in range(n_TPD):
            index = rand_num_TPD[i]
            Y_test_prediction_TPD[index] = Y_test_prediction_TPD_rand[i]

        for i in range(n_FPD):
            index = rand_num_FPD[i]
            Y_test_prediction_FPD[index] = Y_test_prediction_FPD_rand[i]

    else:
        Y_test_prediction_TPD = Y_test_prediction_TPD_rand
        Y_test_prediction_FPD = Y_test_prediction_FPD_rand

    #
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx #

    #% SAVING THE TEST PREDICTIONS IN TEXT FILES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Open files for writing
    # file_TPD = open("Y_TPD_from_python.txt", "w")
    # file_FPD = open("Y_FPD_from_python.txt", "w")

    # # Write in the files
    # for i in range(n_TPD):
    #     str = repr(Y_test_prediction_TPD[i, 0])  # convert variable to string
    #     file_TPD.write(str + "\n")

    # for i in range(n_FPD):
    #     str = repr(Y_test_prediction_FPD[i, 0])  # convert variable to string
    #     file_FPD.write(str + "\n")

    # # Close the files
    # file_TPD.close()
    # file_FPD.close()



# Getting stats of classification accuracies
if data_div_option == 1:  # Holdout method
    train_Accuracy_avg = final_train_Accuracy[0]  # Average accuracy
    train_Accuracy_SD = 0  # Standard deviation of the accuracy

    test_Accuracy_avg = final_test_Accuracy[0]  # Average accuracy
    test_Accuracy_SD = 0  # Standard deviation of the accuracy

    test_Accuracy_TPD_avg = final_test_Accuracy_TPD[0]  # Average accuracy
    test_Accuracy_TPD_SD = 0  # Standard deviation of the accuracy
    test_Accuracy_FPD_avg = final_test_Accuracy_FPD[0]  # Average accuracy
    test_Accuracy_FPD_SD = 0  # Standard deviation of the accuracy

else:  # K-fold method
    train_Accuracy_avg = np.mean(final_train_Accuracy)  # Average accuracy
    train_Accuracy_SD = np.std(final_train_Accuracy)  # Standard deviation of the accuracy

    test_Accuracy_avg = np.mean(final_test_Accuracy)  # Average accuracy
    test_Accuracy_SD = np.std(final_test_Accuracy)  # Standard deviation of the accuracy

    test_Accuracy_TPD_avg = np.mean(final_test_Accuracy_TPD)  # Average accuracy
    test_Accuracy_TPD_SD = np.std(final_test_Accuracy_TPD)  # Standard deviation of the accuracy
    test_Accuracy_FPD_avg = np.mean(final_test_Accuracy_FPD)  # Average accuracy
    test_Accuracy_FPD_SD = np.std(final_test_Accuracy_FPD)  # Standard deviation of the accuracy

# Selection of best trained model
I_max_final = np.argmax(final_test_Accuracy)  # Index of the model with the maximum test accuracy

# Classifier selection
if classifier_option == 1:  # Logistic regression
    I_max_LR = I_max_final
    Accuracy_LR = [train_Accuracy_avg, train_Accuracy_SD, test_Accuracy_avg, test_Accuracy_SD,
                   test_Accuracy_TPD_avg, test_Accuracy_TPD_SD, test_Accuracy_FPD_avg, test_Accuracy_FPD_SD]
elif classifier_option == 2:  # SVM
    I_max_SVM = I_max_final
    Accuracy_SVM = [train_Accuracy_avg, train_Accuracy_SD, test_Accuracy_avg, test_Accuracy_SD,
                    test_Accuracy_TPD_avg, test_Accuracy_TPD_SD, test_Accuracy_FPD_avg, test_Accuracy_FPD_SD]
elif classifier_option == 3:  # Neural Network
    I_max_NN = I_max_final
    Accuracy_NN = [train_Accuracy_avg, train_Accuracy_SD, test_Accuracy_avg, test_Accuracy_SD,
                   test_Accuracy_TPD_avg, test_Accuracy_TPD_SD, test_Accuracy_FPD_avg, test_Accuracy_FPD_SD]
elif classifier_option == 4:  # Random Forest
    I_max_RF = I_max_final
    Accuracy_RF = [train_Accuracy_avg, train_Accuracy_SD, test_Accuracy_avg, test_Accuracy_SD,
                   test_Accuracy_TPD_avg, test_Accuracy_TPD_SD, test_Accuracy_FPD_avg, test_Accuracy_FPD_SD]

print('\nTraining and testing of the algorithm is completed.\n')

print('\nSettings for the algorithm were: ')
print(f'\n\tThreshold multiplier:\n\t\tAccelerometer = {FM_min_SN[0]:.0f}, Acoustic = {FM_min_SN[2]:.0f}, Piezoelectric diaphragm = {FM_min_SN[4]:.0f}.')
print(f'\n\tData division option for training and testing: {data_div_option} \n\t\t(1- holdout; 2,3- K-fold; 4-by participants)')
# print(f'\n\tFPD/TPD ratio in the training data set: {FPD_TPD_ratio:.2f}')
print(f'\n\tCost of getting TPD wrong: {cost_TPD:.2f}')
print(f'\n\tClassifier: {classifier_option} (1- LR; 2- SVM, 3-NN, 4- RF)')

print('Performance of the classifier were: ')
print(f'\n\t Probabilistic Threshold: {threshold_NN:.2f}')
print(f'\n\tTraining accuracy: \n\t\tOverall = {train_Accuracy_avg:.2f}({train_Accuracy_SD:.3f})')
print(f'\n\tTest accuracy: \n\t\tOverall = {test_Accuracy_avg:.2f}({test_Accuracy_SD:.3f}), \n\t\tfor TPD = {test_Accuracy_TPD_avg:.2f}({test_Accuracy_TPD_SD:.3f}),'
      f'\n\t\tfor FPD = {test_Accuracy_FPD_avg:.2f}({test_Accuracy_FPD_SD:.3f}).')

print("\n\nTime taken: ", time.time()- tic)
#%% PERFORMANCE ANALYSIS ===================================================

print('\nPerformance analysis of the algorithm is going on...')
tic = time.time()

# Select the option to perform ROC & PR analysis
ROC_curve_option = 0  # 1: generates ROC & PR curves, 0: doesn't generate ROC & PR curves
if ROC_curve_option == 1:
    # Define parameter for ROC and PR curve generation
    n_division_thd = 7
    thd = np.linspace(0, 0.35, n_division_thd + 1)[1:-1]
    n_ROC_iter = n_division_thd - 1
else:
    thd = [0.50] # ROC curve "thd" parameter is in list, kept it in list as well
    n_ROC_iter = 1

overall_detections = np.zeros((n_ROC_iter, 4))


for k in range(n_ROC_iter):
    print(f'\nIteration for ROC: {k + 1}/{n_ROC_iter}\n')
    
    # ------------ Getting the prediction for the overall data set -----------#
    if classifier_option == 3:
        load_prediction_option = 1  # if 1 load the prediction from the current directory
    else:
        load_prediction_option = 0
        

    if load_prediction_option == 1:
        # Y_test_prediction_TPD = np.loadtxt('Y_TPD_from_python.txt')
        # Y_test_prediction_FPD = np.loadtxt('Y_FPD_from_python.txt')

        prediction_overall_dataset_TPD = Y_test_prediction_TPD >= thd[k]
        prediction_overall_dataset_FPD = Y_test_prediction_FPD >= thd[k]

        prediction_overall_dataset = np.concatenate((prediction_overall_dataset_TPD, prediction_overall_dataset_FPD))

    else:
        for j in range(len(final_test_prediction)):
            test_scores = final_test_scores[j]

            # Calculate the final test prediction
            final_test_prediction[j] = test_scores[:, 1] >= thd[k] #[:, 1] indexes into that array to get the probabilities for class 1

        # Get the prediction of the overall data set based on data_div_option
        if data_div_option == 1:  # Based on holdout method
            # Z = X_norm  # The trained classifier is applied to the whole data set
            if classifier_option == 1:  # Logistic regression
                prediction_overall_dataset = LR_model_selected[I_max_LR].predict(X_norm)

            elif classifier_option == 2:  # SVM
                prediction_overall_dataset = SVM_model_selected[I_max_SVM].predict(X_norm)

            elif classifier_option == 3:  # NN
                prediction_overall_dataset = NN_model_selected[I_max_NN].predict(X_norm)

            elif classifier_option == 4:  # RF
                prediction_overall_dataset = RF_model_selected[I_max_RF].predict(X_norm)

            prediction_overall_dataset_TPD = prediction_overall_dataset[:n_TPD]
            prediction_overall_dataset_FPD = prediction_overall_dataset[n_TPD:]

        elif data_div_option in [2, 3]:  # Based on stratified K-fold division
            prediction_overall_dataset_TPD, prediction_overall_dataset_FPD \
                = get_overall_test_prediction(final_test_prediction, n_TPD, n_data_TPD_each_fold, n_data_TPD_last_fold,
                n_FPD, n_data_FPD_each_fold, n_data_FPD_last_fold, n_iter, rand_num_TPD, rand_num_FPD)

            prediction_overall_dataset = np.concatenate((prediction_overall_dataset_TPD, prediction_overall_dataset_FPD))

        elif data_div_option == 4:  # Based on data division by participants
            prediction_overall_dataset_TPD = np.zeros(1)  # Initialization
            prediction_overall_dataset_FPD = np.zeros(1)
            for i in range(n_participant):
                n_TPD_current = X_TPD_by_participant[i].shape[0]
                n_FPD_current = X_FPD_by_participant[i].shape[0]

                prediction_overall_dataset_TPD = np.concatenate((prediction_overall_dataset_TPD, final_test_prediction[i][:n_TPD_current]))
                prediction_overall_dataset_FPD = np.concatenate((prediction_overall_dataset_FPD, final_test_prediction[i][n_TPD_current:]))

            prediction_overall_dataset_TPD = prediction_overall_dataset_TPD[1:]  # Removing the initialized 1st element
            prediction_overall_dataset_FPD = prediction_overall_dataset_FPD[1:]

            prediction_overall_dataset = np.concatenate((prediction_overall_dataset_TPD, prediction_overall_dataset_FPD))

        else:
            print('\nWrong data division option chosen.\n')
            break

    # Accuracy for the overall prediction
    Accuracy_overall_dataset = np.sum(prediction_overall_dataset == Y) / len(Y)  # Y will be the same as we are using non-randomized data
    Accuracy_overall_dataset_TPD = np.sum(prediction_overall_dataset_TPD == Y[:n_TPD]) / n_TPD
    Accuracy_overall_dataset_FPD = np.sum(prediction_overall_dataset_FPD == Y[n_TPD:]) / n_FPD

    # Getting the detection stats
    matching_index_TPD = 0
    matching_index_FPD = 0
    sensor_data_sgmntd_cmbd_all_sensors_ML = [None] * n_data_files
    
    TPD_indv = [None] * n_data_files
    FPD_indv = [None] * n_data_files
    TND_indv = [None] * n_data_files
    FND_indv = [None] * n_data_files
    
    
    for i in range(n_data_files):
        # Starting notification
        print(f'\tCurrent data file: {i + 1}/{n_data_files}')

        # ~~~~~~~~~~~~~~~~~~~~~~ Segmentation of FM data ~~~~~~~~~~~~~~~~~~~~~%
        # get_segmented_data() function will be used here, which will
        # threshold the data, remove body movement, and dilate the data.
        # Setting for the threshold and dilation are given in the function.
        #   Input variables:  sensor_data- a cell variable;
        #                     min_SN- a vector/ a scalar;
        #                     IMU_map- a column vector
        #                     Fs_sensor, FM_dilation_time- a scalar
        #   Output variables: sensor_data_sgmntd- a cell variable of the same size as the input variable sensor_data_fltd;
        #                     h- a vector

        
        
# ================================ Old Method =================================
        # sensor_data_sgmntd_cmbd_all_sensors, sensor_data_fltd, sensor_data_sgmntd, n_FM_sensors, threshold[i, 0:n_FM_sensors] = get_segmented_data_cmbd_all_sensors(Aclm_data1_fltd[i], Aclm_data2_fltd[i], Acstc_data1_fltd[i], Acstc_data2_fltd[i], Pzplt_data1_fltd[i], Pzplt_data2_fltd[i],
        #                                         FM_min_SN, IMU_map[i], FM_dilation_time, Fs_sensor, sensor_selection)


        # sensor_data_sgmntd_cmbd_all_sensors_labeled = label(sensor_data_sgmntd_cmbd_all_sensors[0])
        # n_label = len(np.unique(sensor_data_sgmntd_cmbd_all_sensors_labeled)) - 1  # Number of labels in sensor_data_cmbd_all_sensors_labeled

        # # ------------------ Mapping the detection by ML ------------------
        # sensor_data_sgmntd_cmbd_all_sensors_ML[i], matching_index_TPD, matching_index_FPD = map_ML_detections(
        #     sensor_data_sgmntd_cmbd_all_sensors_labeled, M_sntn_map[i], n_label,
        #     prediction_overall_dataset_TPD, prediction_overall_dataset_FPD, matching_index_TPD, matching_index_FPD)
# =============================================================================
        # sensor_data_sgmntd_cmbd_all_sensors, sensor_data_fltd, sensor_data_sgmntd, n_FM_sensors, threshold[i, 0:n_FM_sensors] = get_segmented_data_cmbd_all_sensors(Aclm_data1_fltd[i], Aclm_data2_fltd[i], Acstc_data1_fltd[i], Acstc_data2_fltd[i], Pzplt_data1_fltd[i], Pzplt_data2_fltd[i],
        #                                         FM_min_SN, IMU_map[i], FM_dilation_time, Fs_sensor, sensor_selection)
        # user_scheme_labeled, n_label = Sensor_Fusion(sensor_data_sgmntd, sensor_data_sgmntd_cmbd_all_sensors, desired_scheme, data_format)  # taking as numpy array since lable function expects an array
        
        user_scheme = all_data_files_schemes[i][desired_scheme]
        user_scheme_labeled = label(np.array(user_scheme)) # taking as numpy array since lable function expects an array
        n_label = len(np.unique(user_scheme_labeled)) - 1 # Number of labels in the sensor_data_cmbd_all_sensors_labeled
        
        #label() function returs 2D array, we need 1D array for calculaiton.
        user_scheme_labeled = user_scheme_labeled.reshape((user_scheme_labeled.size,)) #(1,n) to (n,) array
        
        # ------------------ Mapping the detection by ML ------------------
        sensor_data_sgmntd_cmbd_all_sensors_ML[i], matching_index_TPD, matching_index_FPD = map_ML_detections(
            user_scheme_labeled, M_sntn_map[i], n_label,
            prediction_overall_dataset_TPD, prediction_overall_dataset_FPD, matching_index_TPD, matching_index_FPD)       


        # ~~~~~~~~~~~~~~~~~ Matching with maternal sensation ~~~~~~~~~~~~~~~~~%
        #   match_with_m_sensation() function will be used here-
        #   Input variables:  sensor_data_sgmntd- A cell variable
        #                                         Each cell contains data from a sensor or a combination.
        #                     sensation_data, IMU_map, M_sntn_Map- cell variables with single cell
        #                     ext_bakward, ext_forward, FM_dilation_time- scalar values
        #                     Fs_sensor, Fs_sensation- scalar values
        #   Output variables: TPD, FPD, TND, FND- vectors with the number of rows = n_data_files
        
        current_ML_detection_map = np.copy([sensor_data_sgmntd_cmbd_all_sensors_ML[i]]) # copy creates an independent array
        TPD_indv[i], FPD_indv[i], TND_indv[i], FND_indv[i] = match_with_m_sensation(
            current_ML_detection_map, sensation_data_trimd[i], IMU_map[i], M_sntn_map[i], ext_backward,
            ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation)
        
        
        
    # ----------------------- Performance analysis ---------------------------#
    # This section will use get_performance_params() function
    #   Input variables:  TPD_all, FPD_all, TND_all, FND_all- single cell/multi-cell variable.
    #                     The number of cells indicates the number of sensor data or combination data provided together.
    #                     Each cell contains a vector with the number of rows = n_data_files
    #
    #   Output variables: SEN_all, PPV_all, SPE_all, ACC_all, FS_all, FPR_all- cell variable with
    #                     size same as the input variables.

    # For individual data sets
    SEN_indv, PPV_indv, SPE_indv, ACC_indv, FS_indv, FPR_indv = get_performance_params(TPD_indv, FPD_indv,TND_indv, FND_indv)
    # All the returned variables are 1x1 cell arrays

    indv_detections = np.concatenate((TPD_indv, FPD_indv, TND_indv, FND_indv), axis=1)

    # For the overall data sets
    TPD_overall = [np.sum(TPD_indv)]
    FPD_overall = [np.sum(FPD_indv)]
    TND_overall = [np.sum(TND_indv)]
    FND_overall = [np.sum(FND_indv)]

    overall_detections[k, :] = [TPD_overall[0], FPD_overall[0], TND_overall[0], FND_overall[0]]
    
    SEN_overall, PPV_overall, SPE_overall, ACC_overall, FS_overall, FPR_overall = get_performance_params(
        TPD_overall, FPD_overall, TND_overall, FND_overall)
    PABAK_overall = 2 * ACC_overall[0] - 1
    
    #ACC > Accuracy ( (TP+TN) / (TP+TN+FP+FN) )
    #PPV > Precision (TP/ (TP+FP) )
    #SPE > Specificity (True Negative Rate) (TN/ (TN+FP) )
    #SEN > Recall (True Positive Rate) (TP/ (TP+FN) )`ACC_indv
    
    detection_stats = [SEN_overall[0], PPV_overall[0], FS_overall, SPE_overall[0], ACC_overall[0], PABAK_overall]

print('\nPerformance analysis is completed.\n')



# ------------------- Displaying performance metrics ----------------------
print('\nSettings for the algorithm were: ')
print(f'\n\tThreshold multiplier:\n\t\tAccelerometer = {FM_min_SN[0]}, Acoustic = {FM_min_SN[2]},'
      f'\n\t\tPiezoelectric diaphragm = {FM_min_SN[4]}.')

# print(f"\n\tData division option for training and testing: {data_div_option}"
#       "\n\t\t(1- holdout; 2,3- K-fold; 4-by participants)")
# print(f"\n\tFPD/TPD ratio in the training data set: {FPD_TPD_ratio:.2f}")
# print(f"\n\tCost of getting TPD wrong: {cost_TPD:.2f}")
# print(f"\n\tClassifier: {classifier_option} (1- LR, 2- SVM, 3-NN, 4- RF)")

print('\nDetection stats:\n\tSEN = %.3f, PPV = %.3f, F1 score = %.3f,' % (SEN_overall[0], PPV_overall[0], FS_overall[0]))
print('\n\tSPE = %.3f, ACC = %.3f, PABAK = %.3f' % (SPE_overall[0], ACC_overall[0], PABAK_overall))

print('\n\tTPD = %.3f, FPD = %.3f, TND = %.3f, FND = %.3f.\n' % (TPD_overall[0], FPD_overall[0], TND_overall[0], FND_overall[0] ))

print("\n\nTime taken: ", time.time()- tic)
# Clearing variable
# del sensor_data_fltd, sensor_data_sgmntd, sensor_data_sgmntd_cmbd_all_sensors, sensor_data_sgmntd_cmbd_all_sensors_labeled


#%%
print("\nCalculating Results of all Segmentation Schemes...")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Result Generation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for scheme_ind, scheme in enumerate(schemes):
    SEN_indv, PPV_indv, SPE_indv, ACC_indv, FS_indv, FPR_indv = get_performance_params(
        TPD_indv[:, scheme_ind], FPD_indv[:, scheme_ind], TND_indv[:, scheme_ind], FND_indv[:, scheme_ind])
    indv_detections = np.concatenate(
        (TPD_indv[:, scheme_ind], FPD_indv[:, scheme_ind], TND_indv[:, scheme_ind], FND_indv[:, scheme_ind]), axis=0)
    TPD_overall = [np.sum(TPD_indv[:, scheme_ind])]
    FPD_overall = [np.sum(FPD_indv[:, scheme_ind])]
    TND_overall = [np.sum(TND_indv[:, scheme_ind])]
    FND_overall = [np.sum(FND_indv[:, scheme_ind])]
    overall_detections[scheme_ind, :] = [ TPD_overall[0], FPD_overall[0], TND_overall[0], FND_overall[0] ]

    SEN_overall, PPV_overall, SPE_overall, ACC_overall, FS_overall, FPR_overall = get_performance_params(
        TPD_overall, FPD_overall, TND_overall, FND_overall)
    PABAK_overall = 2 * ACC_overall[0] - 1
    detection_stats = [SEN_overall[0], PPV_overall[0],
                       FS_overall, SPE_overall[0], ACC_overall[0], PABAK_overall]
    
    SN_current = ','.join(map(str, FM_min_SN))
    result_cell[scenario_count, 0] = f'Scheme {scheme_strs[scheme_ind]}({SN_current})'

    metrics_dict = {
        "SENSITIVITY": SEN_overall[0],
        "PRECISION": PPV_overall[0],
        "SPECIFICITY": SPE_overall[0],
        "ACCURACY": ACC_overall[0],
        "FS_SCORE": FS_overall
    }
    result_cell[scenario_count, 1] = SEN_overall[0]
    result_cell[scenario_count, 2] = PPV_overall[0]
    result_cell[scenario_count, 3] = SPE_overall[0]
    result_cell[scenario_count, 4] = ACC_overall[0]
    result_cell[scenario_count, 5] = FS_overall[0]
    result_cell[scenario_count, 6] = n_Maternal_detected_movement_raw
    result_cell[scenario_count, 7] = n_Maternal_detected_movement_after
    scenario_count = scenario_count + 1
print("Complete\n")

print("Total time ", time.time()-tic) 


## %%Data frame and results saving


df = pd.DataFrame(result_cell, columns=column_names)
# Print the DataFrame

file_path = rf'FMM_analysis_without_ML_6schemes_Dilation_time.csv'

# Save the DataFrame to a CSV file
df.to_csv(file_path, index=False)

# %% ML Prediction
'''
#==================================================================================
#=============================    Prediction   ====================================
#==================================================================================

extracted_data = []

if data_format == '2':
    for i in range(n_data_files):
    # Starting notification
        IMU_map.append(get_IMU_map(IMU_aclm_fltd[i], data_file_names[i], Fs_sensor, data_format))
        # IMU_map[i] = np.arange(0, len(IMU_map[i]), 1)
        #Make IMU map 'False' for this dummy data, because we do not know exact threshold
        #IMU_map = [~arr for arr in IMU_map]

        # print("\tCreating Segmentation of FM data...")

        n_FM_sensors = 4
        sensor_data_fltd = [s1_fltd[i], s2_fltd[i], s5_fltd[i], s6_fltd[i]]
        sensor_data_sgmntd, threshold[i, 0:n_FM_sensors] = get_segmented_data \
        (sensor_data_fltd, FM_min_SN, IMU_map[i], FM_dilation_time, Fs_sensor)
        sensor_data_sgmntd_cmbd_all_sensors = [sensor_data_sgmntd[0] | sensor_data_sgmntd[1] \
        | sensor_data_sgmntd[2] | sensor_data_sgmntd[3]]

segment = 0
sensor_data_sgmntd_cmbd_all_sensors_labeled = label(np.array( sensor_data_sgmntd_cmbd_all_sensors ) ) #taking as numpy array since lable function expects an array
n_label = len(np.unique(sensor_data_sgmntd_cmbd_all_sensors_labeled)) - 1 #Number of labels in the sensor_data_cmbd_all_sensors_labeled

#label() function returs 2D array, we need 1D array for calculaiton.
sensor_data_sgmntd_cmbd_all_sensors_labeled = sensor_data_sgmntd_cmbd_all_sensors_labeled.reshape((sensor_data_sgmntd_cmbd_all_sensors_labeled.size,)) #(1,n) to (n,) array

if n_label: # When there is a detection by the sensor system
    extracted_data.append(extract_detections_modified(sensor_data_fltd, sensor_data_sgmntd, sensor_data_sgmntd_cmbd_all_sensors_labeled, n_label, n_FM_sensors))
segment = segment + n_label
X_extracted, n_extracted = extract_features_modified(extracted_data, threshold,Fs_sensor,n_FM_sensors)

X_extracted_norm, _, _  = normalize_features(X_extracted)


predicted_labels = ML_prediction(data_format, X_extracted_norm, model_filepath)



sensor_data_sgmntd_cmbd_all_sensors_ML = np.zeros_like(sensor_data_sgmntd_cmbd_all_sensors_labeled)

for k in range(1, segment + 1):
    L_min = np.argmax(sensor_data_sgmntd_cmbd_all_sensors_labeled == k)
    L_max = len(sensor_data_sgmntd_cmbd_all_sensors_labeled) - np.argmax(sensor_data_sgmntd_cmbd_all_sensors_labeled[::-1] == k)

    if predicted_labels[k - 1] == 0:
        sensor_data_sgmntd_cmbd_all_sensors_ML[L_min:L_max] = 0
    else:
        sensor_data_sgmntd_cmbd_all_sensors_ML[L_min:L_max] = 1

'''

# %% Plot



#==================================================================================
#================================    Plot   =======================================
#==================================================================================


num_sensors = 6 
file_idx = 0

if len(s1_fltd[file_idx])< Fs_sensor*60:
    x_axis_type_ = 's'
else:
    x_axis_type_ = 'm'

fig, axs = plt.subplots(num_sensors + 2, 1, figsize=(15, 8), sharex=True)

# Plot the combined sensor data
def plotData(axisIdx, data_y, label, ylabel, xlabel, legend=True, xticks=True, plot_type='line', x_axis_type= x_axis_type_):
    # x_axis_type = 's'
    time_each_data_point = np.arange(0, len(data_y), 1)
    if x_axis_type == 'm':
        time_ticks = time_each_data_point / 1024 / 60
    elif x_axis_type == 's':
        time_ticks = time_each_data_point / 1024
    # return time_ticks
    
    # if not plot_type == 'scatter':
    #     axs[axisIdx].plot(time_ticks, data_y, label=label)
    # else:
    #     axs[axisIdx].scatter(time_ticks, data_y, label=label)
    axs[axisIdx].plot(time_ticks, data_y, label=label)  

    axs[axisIdx].set_xlabel(xlabel)
    axs[axisIdx].set_ylabel(ylabel)
    if legend: axs[axisIdx].legend()
    if xticks: axs[axisIdx].set_xticks(np.arange(0, max(time_ticks) + 5, 5))

plotData(0, s1_fltd[file_idx], 'Accelerometer_1', 'Signal', '' )
plotData(1, s2_fltd[file_idx], 'Accelerometer_2', 'Signal', '' )
plotData(2, s3_fltd[file_idx], 'Piezo_1', 'Signal', '' )
plotData(3, s4_fltd[file_idx], 'Piezo_2', 'Signal', '' )
plotData(4, s5_fltd[file_idx], 'Piezo_3', 'Signal', '' )
plotData(5, s6_fltd[file_idx], 'Piezo_4', 'Signal', '' )

# plotData(6, sensor_data_sgmntd_cmbd_all_sensors, 'All Events', 'Detection', '', legend=True )
plotData(6, sensation_data_trimd[file_idx], 'Ground Truth', '', '', legend=True, plot_type='scatter')
plotData(6, IMU_aclm_map[file_idx]*.5, 'IMU_aclm_Map', '', '', legend=True, plot_type='scatter')
# plotData(6, IMU_RPY_map[file_idx]*.6, 'IMU_RPY_map', '', '', legend=True, plot_type='scatter')
# plotData(6, IMU_merged_map[file_idx]*.8, 'IMU_merged_map', '', '', legend=True, plot_type='scatter')

plotData(7, IMU_aclm_fltd[file_idx], 'IMU data', '', '', legend=True, plot_type='scatter')
# plotData(7, sensor_data_sgmntd_cmbd_all_sensors_ML, 'Fetal movement', 'Detection', 'Time(second)', legend=True, xticks=True )

plt.tight_layout()
plt.show()

'''
if not os.path.exists("Results"):
        # Create the directory
        os.makedirs("Results")

file_name_prediction = data_file_path.split("/")[-1].replace(".dat", "")
result_file_path = "Results/prediction_"+file_name_prediction+".png"
if os.path.exists(result_file_path):
        os.remove(result_file_path)
fig.savefig(result_file_path)

total_ML_detection = len(np.unique(label(sensor_data_sgmntd_cmbd_all_sensors_ML))) -1
total_ground_truth = len(np.unique(label(sensation_data_trimd[0]))) -1

# print(f"Total Machine Learning prediction: {total_ML_detection} out of {total_ground_truth}\n")
print(f"Data file duration:\t{round(len(s1_fltd[0])/Fs_sensor, 1)} seconds")
print(f"Total FM ground truth:\t{total_ground_truth}")
print(f"Total ML prediction:\t{total_ML_detection}")
# print(f"Prediction vizualization saved in {result_file_path}\n")
# print("-------------------------------------------------------------------------------------------------\n")
'''


#  aws s3 ls s3://femo-sensor-logfiles