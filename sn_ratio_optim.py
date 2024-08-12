# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 01:07:52 2024

@author: Monaf Chowdhury & Moniruzzaman Akash
For SN Ratio Optimization
"""
# %% Import Libraries
#!/usr/bin/bash python3

import pandas as pd
import numpy as np
from tqdm import tqdm 
#import tkinter as tk
#from tkinter import filedialog
from skimage.measure import label
import glob, pickle, itertools
import os, time, sys, argparse, warnings, logging
from matplotlib import pyplot as plt

from femo import Femo# Local code imports

from skimage.measure import label
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier,\
    VotingClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from scipy.special import expit

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from SP_Functions import load_data_files, get_dataFile_info, get_preprocessed_data,\
    get_info_from_preprocessed_data,get_IMU_map,get_sensation_map,get_segmented_data,\
    match_with_m_sensation,get_performance_params, remove_IMU_from_segmentation, \
    get_segmented_data_cmbd_all_sensors, get_IMU_rot_map, get_merged_map, get_IMU_rot_map_complex, get_IMU_rot_map_TWO
        
from ML_Functions import extract_detections, extract_features, normalize_features,\
    divide_by_holdout, divide_by_K_folds, divide_by_participants, get_prediction_accuracies,\
    projectData, get_overall_test_prediction, map_ML_detections, define_model, ML_prediction
    
from Feature_Ranking import FeatureRanker
from tensorflow.keras.callbacks import EarlyStopping, Callback
import tensorflow as tf
# from aws_server import S3FileManager

warnings.filterwarnings('ignore')
# Change the current working directory to the directory of the current script file
os.chdir(os.path.dirname( os.path.abspath(__file__) ) )

#%% Load Data File Names

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


def get_file_list(folder_path, data_format, participant_folders=False):
    # Initialize an empty list to store file names
    file_list = []
    
    # Check if participant_folders is provided
    if participant_folders:
        # Iterate over each participant folder
        
        for participant_folder in participant_folders:
            participant_path = os.path.join(folder_path, participant_folder)
            if os.path.isdir(participant_path):
                print(participant_folder, "> Found")
                # Get list of files in the participant folder
                files = os.listdir(participant_path)
                
                
                # Iterate over each file
                for file_name in files:
                    # Check the file extension based on data_format
                    if (data_format == '1' and file_name.endswith(".mat")) or \
                       (data_format == '2' and ((file_name.endswith(".dat")) or (file_name.endswith(".csv")))):
                        # If it matches, append the file path to the list
                        file_list.append(folder_path + participant_folder + "/" + file_name)
                    
                
            else:
                print(participant_folder, "> Not Found")
    else:
        # Get list of files in the main folder
        files = os.listdir(folder_path)
        
        # Iterate over each file
        for file_name in files:
            # Check the file extension based on data_format
            if (data_format == '1' and file_name.endswith(".mat")) or \
               (data_format == '2' and ((file_name.endswith(".dat")) or (file_name.endswith(".csv")))):
                # If it matches, append the file path to the list
                # file_list.append(folder_path + participant_folder + "/" + file_name)
                file_list.append(folder_path + file_name)

    print("Total files found:", len(file_list))
    return file_list

def list_all_files(directory):
    file_paths = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Create the full path
        full_path = os.path.join(directory, filename)
        full_path = full_path + '/'
        # Check if it's a file (not a directory)
        files = os.listdir(full_path)
        for file in files:
            final_path = os.path.join(full_path,file)
            file_paths.append(final_path)

    return file_paths

participant_data_in_use = ["F4_12_FA_8D_05_EC",
                            "DC_54_75_C2_23_28",
                            "F4_12_FA_8D_12_D4",
                            "DC_54_75_C0_E8_30",
                            "F4_12_FA_8B_1B_C0",
                            "DC_54_75_C2_22_4C"]

def list_files(directory, mode='IMU_ok'):
    """
    List all files in a directory and its subdirectories.

    Parameters:
    directory (str): The directory to search for files.
    mode (str): The mode of listing files. 'all' for all files, 'IMU_ok' for files in IMU_ok folders.

    Returns:
    list: A list of file paths.
    """
    file_paths = []

    for root, dirs, files in os.walk(directory):
        print (root)
        if mode == 'all':
            for file in files:
                file_paths.append(os.path.join(root, file))
        elif mode == 'IMU_ok':
            if 'IMU_ok' in root:
                for file in files:
                    file_paths.append(os.path.join(root, file))

    return file_paths

# new_data_folder_path = "I:/Other computers/Desktop/WelcomeLeap/Github/FeMo_Analysis/Data_files/"
# new_data_folder_path = "D:/Monaf/New Data FM Monitoring/All_data_files_combined/"
new_data_folder_path = "D:/Monaf/Femo All Data/all_data/A_type"
# new_data_folder_path = "D:/Monaf/New Data FM Monitoring_5_participants_to_csv/"  
# new_data_folder_path = "D:/Monaf/confidential_monaf_only/"
old_data_folder_path = "I:/Other computers/Desktop/WelcomeLeap/Previous Study/All subject data/Fetal movement data/"


if load_single_file:
    data_file_names = [data_file_path]
else:
    if data_format == '1':
        data_file_names = get_file_list(old_data_folder_path, data_format)
    elif data_format == '2':
        data_file_names = list_files(new_data_folder_path, 'IMU_ok') # For drive
        # data_file_names = get_file_list(new_data_folder_path, data_format) # For local machine
data_file_names = list_files(new_data_folder_path, mode='IMU_ok') 


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
FM_min_SN           = [30, 30, 40, 30, 30, 40]  # These values are selected to get SEN of 99%

# 40,40,50,40,40,50
# 70,70,50,30,30,50


IMU_aclm_map = []
IMU_RPY_map = []
IMU_map = []

M_sntn_map =[]
threshold = np.zeros((n_data_files, n_FM_sensors))
TPD_extracted = []
FPD_extracted = []
extracted_TPD_weightage = []
extracted_FPD_weightage = []



original_list = [30, 40, 50, 60, 70, 80, 90]
# original_list = [30, 40]
# Generate all combinations of 6 elements with replacement
# Permutation is 7x7x7 = 349. For aclm take value only 30,40,50. so user should only take first 0-146 elements 
# to reduce computational overhead

all_combinations = list(itertools.product(original_list, repeat=3))
# all_combinations = all_combinations[:147]
# all_combinations = [(30,40,30)]


# aclm_thrs = [0.16,0.18,0.2,0.22,0.24,0.26]
# rot_thrs  = [2,3,4,5,6,6.5]
# from itertools import product
# imu_combinations = list(product(aclm_thrs, rot_thrs))


# scheme_strs = ['Atleast_1_type_of_sensor', 'Atleast_2_type_of_sensor', 'Atleast_3_type_of_sensor', 'Atleast_1_sensor', 'Atleast_2_sensors', 'Atleast_3_sensors', 'Atleast_4_sensors', 'Atleast_5_sensors', 'Atleast_6_sensors']
scheme_strs = ['Atleast_2_type_of_sensor', 'Atleast_3_type_of_sensor', 'Atleast_4_sensors', 'Atleast_5_sensors', 'Atleast_6_sensors']

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
IMU_rot_threshold = 4
IMU_dilation_time = 4

sensor_data_sgmntd_cmbd_all_sensors_list = []
sensor_data_sgmntd_2_type_sensor_list = []
sensor_data_sgmntd_all = []



n_Maternal_detected_movement_raw = 0
n_Maternal_detected_movement_after = 0


for i in range(n_data_files):
    # Starting notification
    print('\nCurrent data file: {}/{}'.format(i+1, n_data_files))

    print("\tCreating IMU Map...")
    IMU_aclm_map.append(get_IMU_map(IMU_aclm_fltd[i], data_file_names[i], Fs_sensor, IMU_aclm_threshold, IMU_dilation_time, data_format))
    # IMU_map[i] = np.arange(0, len(IMU_map[i]), 1)
    # Make IMU map 'False' for this dummy data, because we do not know exact threshold
    # IMU_map = [~arr for arr in IMU_map]
    
    # Creating IMU_rotation map
    IMU_RPY_map.append(get_IMU_rot_map_complex(IMU_rotation_fltd[i], IMU_rot_threshold, IMU_dilation_time, Fs_sensor))
    
    
    IMU_map.append(get_merged_map(IMU_aclm_map[i], IMU_RPY_map[i] ))
    # IMU_map.append( IMU_aclm_map[i] | IMU_RPY_map[i]  )
    
    
    # ----------------------- Creation of M_sensation_map ----------------%
    # get_sensation_map() function is used here. This function gives the
    # sensation map by dilating every detection to past and future. It also
    # revomes the windows that overlaps with IMU_map. Settings for the
    # extension are given inside the function.
    
    print("\tCreating Matarnal Sensation Map...")
    M_sntn_map.append( get_sensation_map(sensation_data_trimd[i], IMU_map[i] ,ext_backward, ext_forward, Fs_sensor, Fs_sensation) )
    
    # -----------------------Calculating the number of maternal sensation detection------
    sensation_data_raw = get_sensation_map(sensation_data_trimd[i], IMU_map[i]*0 ,ext_backward, ext_forward, Fs_sensor, Fs_sensation)
    sensation_data_labeled       = label(sensation_data_raw)
    n_Maternal_detected_movement_raw += len(np.unique(sensation_data_labeled)) - 1 #one label is 0 for all 0s, 1 is deducted to remove the initial value of 0
    
    
    sensation_data_labeled_       = label(M_sntn_map[i])
    n_Maternal_detected_movement_after += len(np.unique(sensation_data_labeled_)) - 1 #one label is 0 for all 0s, 1 is deducted to remove the initial value of 0
    
    

print("Segmentation going on...")

comb_count = 0  # To calculate the combination counter

# ===== Loop for testing set of SN ratios =====
for SN in all_combinations:
    aclm_sm = SN[0]
    pzplt_lrg_sn = SN[1]
    pzplt_sml_sm = SN[2]
    
    
    FM_min_SN = [aclm_sm, aclm_sm, pzplt_lrg_sn, pzplt_sml_sm, pzplt_sml_sm,  pzplt_lrg_sn]
    print(*FM_min_SN, sep = ", ") 
    # print(f"{(scenario_count+1)/10 }/{len(all_combinations)}")

    comb_count = comb_count + 1
    print(f"{comb_count}/{len(all_combinations)}")

    # for imu_thrs in imu_combinations:
    #     IMU_aclm_threshold = imu_thrs[0]
    #     IMU_rot_threshold = imu_thrs[1]
        
    #     print(f"{scenario_count+1}/{len(imu_combinations)}")
    #     print(f"Aclm: {IMU_aclm_threshold} \t Rot: {IMU_rot_threshold}")
        
    #     IMU_aclm_map = []
    #     IMU_RPY_map = []z
    #     IMU_map = []
        
        
    # n_Maternal_detected_movement_raw = 0
    # n_Maternal_detected_movement_after = 0
    

    for i in range(n_data_files):
        # Starting notification
        print('\nCurrent data file: {}/{}'.format(i+1, n_data_files))
    
        # print("\tCreating IMU Map...")
        # IMU_aclm_map.append(get_IMU_map(IMU_aclm_fltd[i], data_file_names[i], Fs_sensor, IMU_aclm_threshold, IMU_dilation_time, data_format))
        # # IMU_map[i] = np.arange(0, len(IMU_map[i]), 1)
        # #Make IMU map 'False' for this dummy data, because we do not know exact threshold
        # #IMU_map = [~arr for arr in IMU_map]
        
        # # Creating IMU_rotation map
        # IMU_RPY_map.append(get_IMU_rot_map(IMU_rotation_fltd[i], IMU_rot_threshold, IMU_dilation_time, Fs_sensor))
        
        
        # IMU_map.append(get_merged_map(IMU_aclm_map[i], IMU_RPY_map[i] ))
        # # A = ( IMU_aclm_map[i] | IMU_RPY_map[i])
        # # IMU_map.append( A  )
        
        
        # # ----------------------- Creation of M_sensation_map ----------------%
        # # get_sensation_map() function is used here. This function gives the
        # # sensation map by dilating every detection to past and future. It also
        # # revomes the windows that overlaps with IMU_map. Settings for the
        # # extension are given inside the function.
        
        # print("\tCreating Matarnal Sensation Map...")
        # M_sntn_map.append( get_sensation_map(sensation_data_trimd[i], IMU_map[i] ,ext_backward, ext_forward, Fs_sensor, Fs_sensation) )
        
        # # -----------------------Calculating the number of maternal sensation detection------
        # sensation_data_raw = get_sensation_map(sensation_data_trimd[i], IMU_map[i]*0 ,ext_backward, ext_forward, Fs_sensor, Fs_sensation)
        # sensation_data_labeled       = label(sensation_data_raw)
        # n_Maternal_detected_movement_raw += len(np.unique(sensation_data_labeled)) - 1 #one label is 0 for all 0s, 1 is deducted to remove the initial value of 0
        
        
        # sensation_data_labeled_       = label(M_sntn_map[i])
        # n_Maternal_detected_movement_after += len(np.unique(sensation_data_labeled_)) - 1 #one label is 0 for all 0s, 1 is deducted to remove the initial value of 0
        
        
        
    
        print("\tCreating Segmentation of FM data...")
        # ---------------------- Segmentation of FM data ---------------------%
        # Here we will threshold the data, remove body movement, and dilate the data.
        # Setting for the threshold and dilation are given in the function.
        # Finally sensor data will be combined using different Fusion Scheme
        
        sensor_data_fltd = [s1_fltd[i], s2_fltd[i], s3_fltd[i], s4_fltd[i], s5_fltd[i], s6_fltd[i]]
        
        #Get segmented map for each sensors
        sensor_data_sgmntd, threshold[i, 0:n_FM_sensors] = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map[i], FM_dilation_time, Fs_sensor)
        
        sensor_data_sgmntd[0] = remove_IMU_from_segmentation(IMU_map[i], sensor_data_sgmntd[0])
        sensor_data_sgmntd[1] = remove_IMU_from_segmentation(IMU_map[i], sensor_data_sgmntd[1])
        sensor_data_sgmntd[2] = remove_IMU_from_segmentation(IMU_map[i], sensor_data_sgmntd[2])
        sensor_data_sgmntd[3] = remove_IMU_from_segmentation(IMU_map[i], sensor_data_sgmntd[3])
        sensor_data_sgmntd[4] = remove_IMU_from_segmentation(IMU_map[i], sensor_data_sgmntd[4])
        sensor_data_sgmntd[5] = remove_IMU_from_segmentation(IMU_map[i], sensor_data_sgmntd[5])
        
        # sensor_data_sgmntd_all.append(sensor_data_sgmntd)
        # Combine sensor segmentation
        sensor_data_sgmntd_cmbd_all_sensors = sensor_data_sgmntd[0] | sensor_data_sgmntd[1] | sensor_data_sgmntd[2] | sensor_data_sgmntd[3] | sensor_data_sgmntd[4] | sensor_data_sgmntd[5]
    
        
        #store segementation list
        # sensor_data_sgmntd_cmbd_all_sensors_list.append(sensor_data_sgmntd_cmbd_all_sensors)
        
        
        
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
    
    
        
    
        # schemes = [sensor_data_sgmntd_atleast_1_type_of_sensor, sensor_data_sgmntd_atleast_2_type_of_sensor, sensor_data_sgmntd_atleast_3_type_of_sensor, sensor_data_sgmntd_atleast_1_sensor,
        #            sensor_data_sgmntd_atleast_2_sensor, sensor_data_sgmntd_atleast_3_sensor, sensor_data_sgmntd_atleast_4_sensor, sensor_data_sgmntd_atleast_5_sensor, sensor_data_sgmntd_atleast_6_sensor]
        
        schemes = [sensor_data_sgmntd_atleast_1_type_of_sensor, sensor_data_sgmntd_atleast_2_type_of_sensor]
        
        # store segementation list        
        # sensor_data_sgmntd_2_type_sensor_list.append(sensor_data_sgmntd_atleast_2_type_of_sensor)
        
        
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
        metrics_dict = {
            "SENSITIVITY": SEN_overall[0],
            "PRECISION": PPV_overall[0],
            "SPECIFICITY": SPE_overall[0],
            "ACCURACY": ACC_overall[0],
            "FS_SCORE": FS_overall[0]
        }
        scheme_strs = ['Atleast_1_type_of_sensor', 'Atleast_2_type_of_sensor']
        column_names = ['Scenario', 'SENSITIVITY', 'PRECISION', 'SPECIFICITY', 'ACCURACY', 'FS_SCORE', 'M_sen_before', 'M_sen_after']
        
        SN_current = FM_min_SN #','.join(map(str, FM_min_SN))
        scenario_count = scenario_count + 1       
        
        result_col = [0] * len(column_names)
        result_col[0] = f'Scheme {scheme_strs[scheme_ind]}({SN_current}) aclm:{IMU_aclm_threshold}_rot:{IMU_rot_threshold}'
        result_col[1] = SEN_overall[0]
        result_col[2] = PPV_overall[0]
        result_col[3] = SPE_overall[0] 
        result_col[4] = ACC_overall[0]
        result_col[5] = FS_overall[0]
        result_col[6] = n_Maternal_detected_movement_raw
        result_col[7] = n_Maternal_detected_movement_after

        
        df = pd.DataFrame([result_col], columns=column_names)

        # file_path = rf'FMM_analysis_with_ML_6schemes_Dilation_time_{scheme_strs[desired_scheme]}.csv'
        csv_file_path = "D:\Monaf\Training_testing\FeMo_Analysis_backup\sn_ratio_optim.csv"

        # Save the DataFrame to a CSV file
        # df.to_csv(file_path, index=False)
          
        # Check if the CSV file exists
        if not os.path.exists(csv_file_path):
            # Create the file and write the header and the first row
            df.to_csv(csv_file_path, index=False)
        else:
            # Append the new row to the existing file
            df.to_csv(csv_file_path, mode='a', header=False, index=False)

        print("Results Generated to 'performance_metrics.csv'")



    
print("Complete\n")
print("Total time ", time.time()-tic) 


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
file_idx = 25



# if len(s1_fltd[file_idx])< Fs_sensor*60:
#     x_axis_type_ = 's'
# else:
#     x_axis_type_ = 'm'
x_axis_type_ = 'm'
# Plot the combined sensor data
def plotData(axisIdx, data_y, label, ylabel, xlabel, legend=False, xticks=True, plot_type='line', x_axis_type= x_axis_type_, color=False, remove_border= True, multiply=0.1):
    # x_axis_type = 's'
    if plot_type == 'line':
        time_each_data_point = np.arange(0, len(data_y), 1)
        if x_axis_type == 'm':
            time_ticks = time_each_data_point / 1024 / 60
        elif x_axis_type == 's':
            time_ticks = time_each_data_point / 1024
    else: #if scatter
        time_each_data_point = np.where(data_y > 0)[0]
        if x_axis_type == 'm':
            time_ticks = time_each_data_point / 1024 / 60
        elif x_axis_type == 's':
            time_ticks = time_each_data_point / 1024
    # return time_ticks
    
    if plot_type == 'line':
        if color == False:
            axs[axisIdx].plot(time_ticks, data_y, label=label, linewidth=3)
        else:
            axs[axisIdx].plot(time_ticks, data_y, label=label, color=color)
    else:
        #if scatter
        data_y = np.ones_like(time_ticks)
        axs[axisIdx].scatter(time_ticks, data_y*multiply, label=label, color='red', marker='*', s=5)
    # axs[axisIdx].plot(time_ticks, data_y, label=label)  

    axs[axisIdx].set_xlabel(xlabel)
    axs[axisIdx].set_ylabel(ylabel)
    # if legend: axs[axisIdx].legend()
    if xticks: axs[axisIdx].set_xticks(np.arange(0, max(time_ticks) + 5, 5))
    
    #Remove Y axis ticks
    axs[axisIdx].set_yticks([])
    
    # Change the xtick font size
    axs[axisIdx].xaxis.set_tick_params(labelsize=15)
    
    #Border removal logic
    if remove_border:
        axs[axisIdx].axis('off')
    else:
        axs[axisIdx].spines['top'].set_visible(False)
        axs[axisIdx].spines['right'].set_visible(False)
        axs[axisIdx].spines['left'].set_visible(False)


fig, axs = plt.subplots(num_sensors + 2, 1, figsize=(15, 8), sharex=True)

start_idx = Fs_sensor*1*60
plotData(0, IMU_aclm_fltd[file_idx], 'IMU data', '', '', legend=False)
plotData(0, IMU_map[file_idx]*3, 'IMU_Map', '', '', legend=False, color="red")


plotData(1, s1_fltd[file_idx], 'Accelerometer_1', 'Signal', '' )
# plotData(1, sensor_data_sgmntd_all[file_idx][0]*np.max(s1_fltd[file_idx])*0.7, '', '', '' )
plotData(1, sensation_data_trimd[file_idx], '', '', '', plot_type='scatter', multiply=np.max(s1_fltd[file_idx])*0.7)


plotData(2, s2_fltd[file_idx], 'Accelerometer_2', 'Signal', '' )
# plotData(2, sensor_data_sgmntd_all[file_idx][1]*np.max(s2_fltd[file_idx])*0.7, '', '', '' )
plotData(2, sensation_data_trimd[file_idx], '', '', '', plot_type='scatter', multiply=np.max(s2_fltd[file_idx])*0.7 )


plotData(3, s3_fltd[file_idx], 'Piezo_1', 'Signal', '' )
# plotData(3, sensor_data_sgmntd_all[file_idx][2]*np.max(s3_fltd[file_idx])*0.7, '', '', '' )
plotData(3, sensation_data_trimd[file_idx], '', '', '', plot_type='scatter' , multiply=np.max(s3_fltd[file_idx])*0.7)

plotData(4, s4_fltd[file_idx], 'Piezo_2', 'Signal', '' )
# plotData(4, sensor_data_sgmntd_all[file_idx][3]*np.max(s4_fltd[file_idx])*0.14, '', '', '' )
plotData(4, sensation_data_trimd[file_idx], '', '', '', plot_type='scatter', multiply=np.max(s4_fltd[file_idx])*0.2 )

plotData(5, s5_fltd[file_idx], 'Piezo_3', 'Signal', '' )
# plotData(5, sensor_data_sgmntd_all[file_idx][4]*np.max(s5_fltd[file_idx])*0.7, '', '', '' )
plotData(5, sensation_data_trimd[file_idx], '', '', '', plot_type='scatter' , multiply=np.max(s5_fltd[file_idx])*0.7)

plotData(6, s6_fltd[file_idx], 'Piezo_4', 'Signal', '' )
# plotData(6, sensor_data_sgmntd_all[file_idx][5]*np.max(s6_fltd[file_idx])*0.7, '', '', '' )
plotData(6, sensation_data_trimd[file_idx], '', '', '', plot_type='scatter' , multiply=np.max(s6_fltd[file_idx])*0.7)


plotData(7, sensor_data_sgmntd_2_type_sensor_list[file_idx], 'All Events', 'Detection', '', color= 'green', remove_border=False)
# plotData(7, M_sntn_map[file_idx]*0.6, 'Ground Truth', '', 'Time(minutes)', legend=False, plot_type='line', remove_border=False)
plotData(7, sensation_data_trimd[file_idx], 'Ground Truth', '', '', 'Time(minutes)', plot_type='scatter', remove_border=False, multiply= 0.6)



# plotData(6, IMU_RPY_map[file_idx]*.6, 'IMU_RPY_map', '', '', legend=True, plot_type='scatter')

# plotData(7, sensor_data_sgmntd_cmbd_all_sensors_ML, 'Fetal movement', 'Detection', 'Time(second)', legend=True, xticks=True )

plt.tight_layout()
plt.show()
# fig.savefig(f"Data_files/Plots/{file_idx}.png")
# plt.close(fig)
# print(file_idx)

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
