# %% Import Libraries
#!/usr/bin/bash python3

import pandas as pd
import numpy as np
#import tkinter as tk
#from tkinter import filedialog
from skimage.measure import label
import glob
import os, time, sys, argparse, warnings, logging
from matplotlib import pyplot as plt

from femo import Femo# Local code imports

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from SP_Functions import load_data_files, get_preprocessed_data, get_IMU_map, get_IMU_rot_map, get_segmented_data, \
    get_sensation_map, match_with_m_sensation, get_performance_params, get_merged_map
        
from ML_Functions import extract_detections, extract_detections_modified, extract_features_modified, normalize_features, ML_prediction

from aws_server import S3FileManager

warnings.filterwarnings('ignore')
# Change the current working directory to the directory of the current script file
os.chdir(os.path.dirname( os.path.abspath(__file__) ) )


# model_filepath = "Model_folder/rf_model_selected_two_sensor.pkl"
# data_file_path = "Data_files/F4_12_FA_8D_05_EC/log_2024_04_05_18_24_11.dat" # Data file to use when no data file argument given
data_file_path = "Data_files/log_2024_05_09_10_44_54_IMU_thresh_chk.dat"
load_single_file = True

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

new_data_folder_path = "I:/Other computers/Desktop/WelcomeLeap/Github/FeMo_Analysis/Data_files/F4_12_FA_8D_05_EC/"
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
ext_forward         = 2.0  # Forward extension length in second
FM_dilation_time    = 3 # Dilation size in seconds, its the minimum size of a fetal movement
n_FM_sensors        = 6   # Number of FM sensors
FM_min_SN           = [70,70,50,30,30,50]  # These values are selected to get SEN of 99%




IMU_map = []
IMU_RPY_map = []
IMU_merged_map = []

M_sntn_map =[]
threshold = np.zeros((n_data_files, n_FM_sensors))
TPD_extracted = []
FPD_extracted = []
extracted_TPD_weightage = []
extracted_FPD_weightage = []


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
IMU_rot_threshold = 5
IMU_dilation_time = 4


print("Segmentation going on...")

# ===== Loop for testing set of SN ratios =====
# for SN in all_combinations:
#     aclm_sm = SN[0]
#     pzplt_lrg_sn = SN[1]
#     pzplt_sml_sm = SN[2]
    
    
#     FM_min_SN = [aclm_sm, aclm_sm, pzplt_lrg_sn, pzplt_sml_sm, pzplt_sml_sm,  pzplt_lrg_sn]
#     print(*FM_min_SN, sep = ", ") 
#     print(f"{scenario_count+1}/{len(all_combinations)}")
    
n_Maternal_detected_movement_raw = 0
n_Maternal_detected_movement_after = 0
    
for i in range(n_data_files):
    # Starting notification
    print('\nCurrent data file: {}/{}'.format(i+1, n_data_files))

    print("\tCreating IMU Map...")
    IMU_map.append(get_IMU_map(IMU_aclm_fltd[i], data_file_names[i], Fs_sensor, IMU_aclm_threshold, IMU_dilation_time, data_format))
    # IMU_map[i] = np.arange(0, len(IMU_map[i]), 1)
    #Make IMU map 'False' for this dummy data, because we do not know exact threshold
    #IMU_map = [~arr for arr in IMU_map]
    
    # Creating IMU_rotation map
    IMU_RPY_map.append(get_IMU_rot_map(IMU_rotation_fltd[i], IMU_rot_threshold, IMU_dilation_time, Fs_sensor))
    
    
    IMU_merged_map.append(get_merged_map(IMU_map[i], IMU_RPY_map[i] ))
    
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

print("Total time ", time.time()-tic) 
# %% 
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
# plotData(6, sensation_data_trimd[file_idx], 'Ground Truth', '', '', legend=True, plot_type='scatter')
plotData(6, IMU_map[file_idx]*.5, 'IMU_aclm_Map', '', '', legend=True, plot_type='scatter')
plotData(6, IMU_RPY_map[file_idx]*.6, 'IMU_RPY_map', '', '', legend=True, plot_type='scatter')
plotData(6, IMU_merged_map[file_idx]*.8, 'IMU_merged_map', '', '', legend=True, plot_type='scatter')

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