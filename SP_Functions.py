# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re #import regular expression for extracting file names
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, zpk2sos, sosfiltfilt
from scipy.ndimage import binary_dilation
import concurrent.futures
from skimage.measure import label
from femo import Femo
#from tkinter import filedialog
#import tkinter as tk
import os, sys, glob
import time
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


def custom_binary_dilation(data, dilation_size):
    """
    author: @mnakash
    
    Perform binary dilation on input binary data using a dilate with binary structure of size dilation_size.
    Dilation will be performed in both direction with half of the dilation_size.
    
    Example:
        Input   data = np.array([0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,0,0])
                dilation_size = 3
        Output  dilated_data  = np.array([0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0])
        
    Parameters:
    - data: Binary input data (1D NumPy array)
    - dilation_size: Binary structuring element size (int)

    Returns:
    - dilated_data (1D NumPy array)
    """
    
    
    labelled = label(data)#get number of spikes labelling them in ascending order
    interested_index = [] #to hold indices where to perform dilation
    
    
    for i in range(1, max(labelled)+1): #from 1 to total no_of_spikes
        indexs = np.where(labelled == i)[0] #indexs where i is found
        if len(indexs)>1:
            #store only firts and last index
            interested_index.append(indexs[0])
            interested_index.append(indexs[-1])
        else:
            interested_index.append(indexs[0])
    
    
    dilate_before = dilation_size//2
    dilate_after = dilation_size-dilate_before -1 #minus 1 to exclude current index
    
    dilated_data = data.copy()
    for i in interested_index:
        start_idx = max(0, i - dilate_before)
        end_idx = min(i + dilate_after, len(data)-1)
        
        dilated_data[start_idx:end_idx+1] = 1

    return dilated_data


def load_data_files(data_format, data_file_names):
    # Count the number of data files to be loaded
    # data_file_names is a tuple of single element in case of a single data file selected.
    # Otherwise, it is a tuple of variable containing 1 name in each element
    
    #Get the number of subject data file selected

    # For old data: 
    # s1 is aclm1, # s2 is aclm2, # s3 is acstc1, # s4 is acstc2, # s5 is pzplt1, # s6 is pzplt2
    # sampling rate for all sensor is 1024

    # For new data: 
    # s1 is aclm1, # s2 is aclm2, # s3 is pzplt1, # s4 is pzplt2, # s5 is pzplt3, # s6 is pzplt4
    # New data do not have flexi data, so we have passed a blank array for flexi data to keep the same format of return values
    # sampling rate for piezo is 1024, accelrometer 512, imu 256. We have resampled all sensor data using upsampling technique.
    # To do upsampling, we have used linear interpolation technique
    
    # Since sensor combination on old and new belt is different, generalized variables s1, s2,....s6 are used
    
    #If no data file selected, exit.
    if not len(data_file_names):
        print("No Data file selected")
        sys.exit()
    # tic = time.time()

    n_data_files = len(data_file_names)
    
    # Notify start of the step
    print("\n#Data file loading is going on...")
    
    
    if data_format == '1': 
        print("Loading Old Belt Data")
        # Locate and load the data files
        # root = tk.Tk()
        # root.attributes("-topmost", True)
        # root.withdraw()
        # try:
        #     data_file_names = filedialog.askopenfilenames(title="Select the data files", filetypes=(("Mat Files", "*.mat"), )) #Returns file names with the extension in a tuple
        # except:
        #     print("Some error occured")
        #     exit(1)
        # root.destroy()
        

        # Variable declaration
        #Declare 1D array for filled with zero for each variable
        # sensation_data_SD1 = []
        # sensation_data_SD2 = []
        Acstc_data1 = []
        Acstc_data2 = []
        Aclm_data1 = []
        Aclm_data2 = []
        Pzplt_data1 = []
        Pzplt_data2 = []
        Flexi_data = []
        IMU_aclm = []
        sensation_data = []

        # Loading the data files
        with tqdm(total=n_data_files) as pbar:
            for i in range(n_data_files):
                # print('\n\tCurrent data file: %.0f/%.0f',i,n_data_files)
                
                data = loadmat(data_file_names[i], squeeze_me=True)
    
                sensor_data_SD1 = data["data_var"][0]
                sensor_data_SD2 = data["data_var"][2]
                sensation_data.append(data["data_var"][1]) #sensation_data_SD1
                
                # sensation_data_SD2.append(data["data_var"][3])
    
                # Extracting sensor data
                Flexi_data.append(sensor_data_SD1[:, 0])  # Force sensor data to measure tightness of the belt
                Pzplt_data1.append(sensor_data_SD1[:, 1]) # Left Piezo-plate data to measure abdominal vibration
                Pzplt_data2.append(sensor_data_SD1[:, 2]) # Right Piezo-plate data to measure abdominal vibration
                Acstc_data1.append(sensor_data_SD1[:, 3]) # Left Acoustic sensor data to measure abdominal vibration
                IMU_aclm_xyz = sensor_data_SD1[:, 4:7]  # Accelerometer x,y and z axis data to measure maternal body movement
    
                Acstc_data2.append(sensor_data_SD2[:, 0]) # Right Acoustic sensor data to measure abdominal vibration
                Aclm_data2_xyz = sensor_data_SD2[:, 1:4]  # Right Accelerometer x,y and z axis data to measure abdominal vibration
                Aclm_data1_xyz = sensor_data_SD2[:, 4:7]  # Left Accelerometer  x,y and z axis data to measure abdominal vibration
            
    
                # Resultant of acceleration
                IMU_aclm.append(np.sqrt(np.sum(IMU_aclm_xyz ** 2, axis=1)))
                Aclm_data1.append(np.sqrt(np.sum(Aclm_data1_xyz ** 2, axis=1)))
                Aclm_data2.append(np.sqrt(np.sum(Aclm_data2_xyz ** 2, axis=1)))
                
                # Update the progress bar
                pbar.update(1)

        s1 = Aclm_data1 
        s2 = Aclm_data2 
        s3 = Acstc_data1 
        s4 = Acstc_data2 
        s5 = Pzplt_data1 
        s6 = Pzplt_data2 
        IMU_rotation= []
        IMU_mag     = []
        
    elif data_format == '2':
        print("Loading New belt Data")
        # Open a file dialog for selecting files
    
        # root = tk.Tk()
        # root.withdraw()  # Hide the main window
    
        # file_path = filedialog.askopenfilename(title="Select Data File", filetypes=[("DAT files", "*.dat")])
    
        # Check if a file was selected
        # if file_path:
        #     read_data_file = file_path
        #     data_file_names = read_data_file
        # else:
        #     print("No file selected. Exiting.")
                
        Aclm_data1  = []
        Aclm_data2  = []
        Pzplt_data1 = []
        Pzplt_data2 = []
        Pzplt_data3 = []
        Pzplt_data4 = []
        Flexi_data  = []
        IMU_aclm    = []
        IMU_rotation= []
        IMU_mag     = []
        sensation_data = []
        
        
        # Loading the data files
        with tqdm(total=n_data_files) as pbar:
            for i in range(n_data_files):
                
                read_data = Femo(data_file_names[i])
            
                all_sensor_df = (read_data.dataframes["piezos"]
                                .join(read_data.dataframes["accelerometers"])
                                .join(read_data.dataframes["imu"])
                                .join(read_data.dataframes["force"])
                                .join(read_data.dataframes["push_button"])
                                .join(read_data.dataframes["timestamp"]))
            
            
                # Resample accelerometer data using linear interpolation
            
                ### Accelerometer 1
                all_sensor_df['x1'] = pd.Series(all_sensor_df['x1']).interpolate(method='linear', limit_direction='both')
                all_sensor_df['y1'] = pd.Series(all_sensor_df['y1']).interpolate(method='linear', limit_direction='both')
                all_sensor_df['z1'] = pd.Series(all_sensor_df['z1']).interpolate(method='linear', limit_direction='both')
            
                ### Accelerometer 2
                all_sensor_df['x2'] = pd.Series(all_sensor_df['x2']).interpolate(method='linear', limit_direction='both')
                all_sensor_df['y2'] = pd.Series(all_sensor_df['y2']).interpolate(method='linear', limit_direction='both')
                all_sensor_df['z2'] = pd.Series(all_sensor_df['z2']).interpolate(method='linear', limit_direction='both')
            
                ### IMU_data
                all_sensor_df['rotation_r'] = pd.Series(all_sensor_df['rotation_r']).interpolate(method='linear', limit_direction='both')
                all_sensor_df['rotation_i'] = pd.Series(all_sensor_df['rotation_i']).interpolate(method='linear', limit_direction='both')
                all_sensor_df['rotation_j'] = pd.Series(all_sensor_df['rotation_j']).interpolate(method='linear', limit_direction='both')
                all_sensor_df['rotation_k'] = pd.Series(all_sensor_df['rotation_k']).interpolate(method='linear', limit_direction='both')
                all_sensor_df['accel_x']    = pd.Series(all_sensor_df['accel_x']).interpolate(method='linear', limit_direction='both')
                all_sensor_df['accel_y']    = pd.Series(all_sensor_df['accel_y']).interpolate(method='linear', limit_direction='both')
                all_sensor_df['accel_z']    = pd.Series(all_sensor_df['accel_z']).interpolate(method='linear', limit_direction='both')
                all_sensor_df['magnet_x']   = pd.Series(all_sensor_df['magnet_x']).interpolate(method='linear', limit_direction='both')
                all_sensor_df['magnet_y']   = pd.Series(all_sensor_df['magnet_y']).interpolate(method='linear', limit_direction='both')
                all_sensor_df['magnet_z']   = pd.Series(all_sensor_df['magnet_z']).interpolate(method='linear', limit_direction='both')
            
            
                selected_data_columns = ['p1', 'p2', 'p3', 'p4', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'rotation_r', 'rotation_i', 'rotation_j', 'rotation_k', 'magnet_x','magnet_y','magnet_z', 'accel_x', 'accel_y', 'accel_z', 'button']
                selected_sensor_data = all_sensor_df[selected_data_columns]
                
                IMU_columns = ['rotation_r', 'rotation_i', 'rotation_j', 'rotation_k', 'magnet_x','magnet_y','magnet_z', 'accel_x', 'accel_y', 'accel_z']
                FM_sensor_columns = ['p1', 'p2', 'p3', 'p4', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']
                
                # Convert them to voltage value
                # All sensor is 16 bit data and max voltage is 3.3v
                #  Voltage = (Raw data / 2^ADC resolution) * max_voltage
            
                max_sensor_value = 2**16 - 1  
                max_voltage = 3.3  
        
                for column in selected_sensor_data.columns:
                    if column in FM_sensor_columns:
                        selected_sensor_data.loc[:, column] = (selected_sensor_data[column] / max_sensor_value) * max_voltage
            
                # •	contains the rotation vector, which is the most accurate position (based on magnetometer, accelerometer and gyroscope)
                # •	rotation vector is currently in quaternion format
                # •	rotation vector originally is a float variable, it is stored as int16 with the following conversion: 
                #       rounded (original_float_value x 10000)
                # •	Magnetic vector is originally in Microtesla
                #       Magnetic vector is stored as int16: rounded (original_float x 100)
                # •	Linear acceleration is originally in m/s^2 (gravity excluded)
                #       Linear acceleration is stored as int16: rounded (original_float_value x 1000)
                
                IMU_all = selected_sensor_data[['rotation_r','rotation_i','rotation_j', 'rotation_k',
                                                  'magnet_x','magnet_y','magnet_z',
                                                  'accel_x','accel_y','accel_z']]
                
                IMU_aclm_single_file    = (IMU_all[['accel_x', 'accel_y', 'accel_z']]/1000)
                IMU_mag_single_file     = (IMU_all[['magnet_x', 'magnet_y', 'magnet_z']]/100)
                IMU_rotation_quat       = (IMU_all[['rotation_i', 'rotation_j', 'rotation_k', 'rotation_r']]/10000)
                
                
                
                # --------- Quaternion to euler conversion is not possible with zero-magnitude rows -----
                # --------- Converting zero-magnitude rows with the first nonzero-magnitude row values --
                # Find the first non-zero orientation row
                non_zero_row = IMU_rotation_quat[(IMU_rotation_quat != 0).any(axis=1)].iloc[0].tolist()
                # Replace rows with all zeros with first valid nonzero-magnitude row
                IMU_rotation_quat.loc[(IMU_rotation_quat == 0).all(axis=1)] = non_zero_row
                
                
                IMU_rotation_quat = IMU_rotation_quat.values #convert to numpy array
                
                
                rotation = R.from_quat(IMU_rotation_quat)
                IMU_rotation_rpy = rotation.as_euler('xyz', degrees=True)                
                IMU_rotation_rpy = pd.DataFrame(IMU_rotation_rpy, columns=['roll', 'pitch', 'yaw'])
                
                
                # Function to convert quaternion to euler angles
                # def quaternion_to_euler(row):
                    
                #     try:
                #         quat = [row['rotation_r'], row['rotation_i'], row['rotation_j'], row['rotation_k']]
                #         rotation = R.from_quat(quat)
                #         euler = rotation.as_euler('xyz', degrees=True)
                #     except:
                #         euler = np.array([0,0,0])
                #     return pd.Series(euler, index=['roll', 'pitch', 'yaw'])
                
                # print("Start")
                # # Apply the function to each row of the DataFrame
                # IMU_rotation_rpy = IMU_rotation_quat.apply(quaternion_to_euler, axis=1)
                # print("End")
                
                
                
                Pzplt_data1.append( np.array(selected_sensor_data['p1']) )
                Pzplt_data2.append( np.array(selected_sensor_data['p2']) )
                Pzplt_data3.append( np.array(selected_sensor_data['p3']) )
                Pzplt_data4.append( np.array(selected_sensor_data['p4']) )
            
                # Calculate magnitude values for FM accelerometer data
                Aclm_data1.append( np.linalg.norm(selected_sensor_data[['x1', 'y1', 'z1' ]], axis=1) )
                Aclm_data2.append( np.linalg.norm(selected_sensor_data[['x2', 'y2', 'z2' ]], axis=1) )
            
                # Calculate magnitude values for IMU accelerometers
                IMU_aclm.append( np.linalg.norm(IMU_aclm_single_file[['accel_x', 'accel_y', 'accel_z']], axis=1) )
                IMU_mag.append( np.linalg.norm(IMU_mag_single_file[['magnet_x', 'magnet_y', 'magnet_z']], axis=1) )
                
                IMU_rotation.append(IMU_rotation_rpy) # Rotation data is not combined
                
                
                # New data do not have flexi data, so we have passed a blank array for flexi data to keep the same format of return values
                Flexi_data.append(np.zeros(len(Pzplt_data1)))
            
                #Get maternal sensation 
                sensation_data.append( np.array(selected_sensor_data['button']) )
                
                
                # Update the progress bar
                pbar.update(1)
    
        s1 = Aclm_data1 
        s2 = Aclm_data2 
        s3 = Pzplt_data1 
        s4 = Pzplt_data2 
        s5 = Pzplt_data3 
        s6 = Pzplt_data4

    return s1, s2, s3, s4, s5, s6, Flexi_data, IMU_aclm, IMU_rotation, IMU_mag, sensation_data


def getParticipantID(dataFileName):
    """
    Parameters
    ----------
    dataFileName : String 
        single data file name with/without path.
        i.e. "C:/parent/child/S1_Day3_dataFile_000.mat"

    Returns
    -------
    participantID : String
        subject ID of the participant
        i.e '1'
    """

    #Get file name without path
    DF_names = dataFileName.split("/")[-1]
    
    #Use regular expression to get subject ID(digits followed by "S") )
    participant_ID = re.search(r"S(\d+)", DF_names)
    if participant_ID:
        participant_ID = participant_ID.group(1)
    else:
        raise Exception("This Data file has naming inconsistency")
    
    return participant_ID


def get_dataFile_info(data_file_names, Fs_sensor, sample_data):
    # Calculate the total number of data files
    n_data_files = len(sample_data)

    # Calculate the number of data files from different participants
    n_DF_P1, n_DF_P2, n_DF_P3, n_DF_P4, n_DF_P5 = 0, 0, 0, 0, 0
    for i in range(n_data_files):
        
        participant_ID = getParticipantID(data_file_names[i])

        if participant_ID == '1':
            n_DF_P1 += 1
        elif participant_ID == '2':
            n_DF_P2 += 1
        elif participant_ID == '3':
            n_DF_P3 += 1
        elif participant_ID == '4':
            n_DF_P4 += 1
        elif participant_ID == '5':
            n_DF_P5 += 1
        else:
            raise Exception("This Data file has naming inconsistency")

    # Calculating the duration of each data file
    duration_raw_data_files = []
    for i in range(n_data_files):
        duration_raw_data_files.append(len(sample_data[i]) / (Fs_sensor * 60))  # Duration in minutes

    return (
        n_data_files,
        n_DF_P1,
        n_DF_P2,
        n_DF_P3,
        n_DF_P4,
        n_DF_P5,
        duration_raw_data_files,
    )


def get_preprocessed_data(s1, s2, s3, s4, s5, s6, Flexi_data, IMU_aclm, IMU_rotation, sensation_data, Fs_sensor, data_format):
    
    #GET_PREPROCESSED_DATA Summary of this function goes here
    #   Detailed explanation goes here
    
    # Pre-processing steps includes filtering/detrending and trimming the data
    # Triming is done to remove some initial and final unwanted data
    # Force data analysis is also included in this step


    # Variable declaration
    s1_fltd = s1.copy()
    s2_fltd = s2.copy()
    s3_fltd = s3.copy()
    s4_fltd = s4.copy()
    s5_fltd = s5.copy()
    s6_fltd = s6.copy()
    Flexi_data_fltd     = Flexi_data.copy()
    IMU_aclm_fltd       = IMU_aclm.copy()
    IMU_rotation_fltd   = IMU_rotation.copy()
    sensation_data_trimd = sensation_data.copy()

    n_data_files = len(s1) # Number of data files in the passed data set
    
    
    
    # ---------------------------- Filter design -----------------------------#
    #3 types of filters are designed-
    #bandpass filter, low-pass filter, and IIR notch filter.


    # Filter setting
    filter_order = 10
    lowCutoff_FM = 1
    highCutoff_FM = 30
    lowCutoff_IMU = 1
    highCutoff_IMU = 10
    highCutoff_force = 10

    # Trim settings
    if data_format == '1':
        start_removal_period = 30  # Removal period in seconds
        end_removal_period = 30  # Removal period in seconds

    elif data_format == '2':
        
        if len(s1[0])> Fs_sensor*5*60: #If greater than 5 minutes remove last and first 30 seconds
            start_removal_period = 30  # Removal period in seconds
            end_removal_period = 30  # Removal period in seconds
        
        else:                                   #Else remove just 5 seconds
            start_removal_period = 5  # Removal period in seconds
            end_removal_period = 5  # Removal period in seconds
    
    # Starting notification
    print('\n#Settings for pre-processing and trimming-')
    print(f'\tFilter order: {filter_order:.1f}')
    print(f'\tIMU band-pass: {lowCutoff_IMU}-{highCutoff_IMU} Hz')
    print(f'\tFM band-pass: {lowCutoff_FM}-{highCutoff_FM} Hz')
    print(f'\tForce sensor low-pass: {highCutoff_force} Hz')
    print("\tTrim settings:")
    print(f'\t\tAt the start: {start_removal_period} s')
    print(f'\t\tAt the end: {end_removal_period} s')



    # ================Bandpass filter design==========================
    #   A band-pass filter with a passband of 1-20 Hz is disigned for the fetal fetal movement data
    #   Another band-pass filer with a passband of 1-10 Hz is designed for the IMU data

    # ========SOS-based design
    #Get second-order sections form
    sos_FM  = butter(filter_order / 2, np.array([lowCutoff_FM, highCutoff_FM]) / (Fs_sensor / 2), 'bandpass', output='sos')# filter order for bandpass filter is twice the value of 1st parameter
    sos_IMU = butter(filter_order / 2, np.array([lowCutoff_IMU, highCutoff_IMU]) / (Fs_sensor / 2), 'bandpass', output='sos')
    
    # ========Zero-Pole-Gain-based design
    # z_FM, p_FM, k_FM = butter(filter_order / 2, np.array([lowCutoff_FM, highCutoff_FM]) / (Fs_sensor / 2), 'bandpass', output='zpk')# filter order for bandpass filter is twice the value of 1st parameter
    # sos_FM, g_FM = zpk2sos(z_FM, p_FM, k_FM) #Convert zero-pole-gain filter parameters to second-order sections form
    # z_IMU,p_IMU,k_IMU = butter(filter_order / 2, np.array([lowCutoff_IMU, highCutoff_IMU]) / (Fs_sensor / 2), 'bandpass', output='zpk')
    # sos_IMU, g_IMU = zpk2sos(z_IMU,p_IMU,k_IMU)
    
    # ========Transfer function-based desing
    #Numerator (b) and denominator (a) polynomials of the IIR filter
    # b_FM,a_FM   = butter(filter_order/2,np.array([lowCutoff_FM, highCutoff_FM])/(Fs_sensor/2),'bandpass', output='ba')# filter order for bandpass filter is twice the value of 1st parameter
    # b_IMU,a_IMU = butter(filter_order/2,np.array([lowCutoff_IMU, highCutoff_IMU])/(Fs_sensor/2),'bandpass', output='ba')
    


    # =================Low-pass filter design==========================
    # This filter is used for the force sensor data only
    
    # ========SOS-based design
    #Get second-order sections form
    sos_force = butter(filter_order, highCutoff_force / (Fs_sensor / 2), 'low', output='sos')
    
    # ========Zero-Pole-Gain-based design
    # z_force,p_force,k_force = butter(filter_order, highCutoff_force / (Fs_sensor / 2), 'low', output='zpk')
    # sos_force,g_force = zpk2sos(z_force,p_force,k_force)
    
    # ========Transfer function-based desing
    #Numerator (b) and denominator (a) polynomials of the IIR filter
    # b_force,a_force = butter(filter_order, highCutoff_force / (Fs_sensor / 2), 'low', output='ba')
    
    
    # =============== Loop for filtering and trimming the data ================
    
    #add progress bar with tqdm
    with tqdm(total=n_data_files) as pbar:
        for i in range(n_data_files):
            # print('\n\tCurrent data file: {}/{}'.format(i+1, n_data_files))
    
            # -----------------------Data filtering--------------------------------
            
            # Bandpass filtering
            s1_fltd[i]       = sosfiltfilt(sos_FM,  s1_fltd[i])
            s2_fltd[i]       = sosfiltfilt(sos_FM,  s2_fltd[i])
            s3_fltd[i]       = sosfiltfilt(sos_FM,  s3_fltd[i])
            s4_fltd[i]       = sosfiltfilt(sos_FM,  s4_fltd[i])
            s5_fltd[i]       = sosfiltfilt(sos_FM,  s5_fltd[i])
            s6_fltd[i]       = sosfiltfilt(sos_FM,  s6_fltd[i])
            IMU_aclm_fltd[i] = sosfiltfilt(sos_IMU, IMU_aclm_fltd[i])
            IMU_rotation_fltd[i]['roll']    = sosfiltfilt(sos_IMU, IMU_rotation_fltd[i]['roll'].values)
            IMU_rotation_fltd[i]['pitch']   = sosfiltfilt(sos_IMU, IMU_rotation_fltd[i]['pitch'].values)
            IMU_rotation_fltd[i]['yaw']     = sosfiltfilt(sos_IMU, IMU_rotation_fltd[i]['yaw'].values)
            
            
            
            if data_format == '1':
                # Low-pass filtering
                Flexi_data_fltd[i]  = sosfiltfilt(sos_force, Flexi_data_fltd[i])
            
            
            # -----------------------Data trimming---------------------------------
            
            # Trimming of raw data
            start_index = start_removal_period * Fs_sensor
            end_index = -end_removal_period * Fs_sensor
            sensation_data_trimd[i] = sensation_data_trimd[i][start_index:end_index]
            
            # Trimming of filtered data
            s1_fltd[i]          = s1_fltd[i][start_index:end_index]
            s2_fltd[i]          = s2_fltd[i][start_index:end_index]
            s3_fltd[i]          = s3_fltd[i][start_index:end_index]
            s4_fltd[i]          = s4_fltd[i][start_index:end_index]
            s5_fltd[i]          = s5_fltd[i][start_index:end_index]
            s6_fltd[i]          = s6_fltd[i][start_index:end_index]
            Flexi_data_fltd[i]  = Flexi_data_fltd[i][start_index:end_index]
            IMU_aclm_fltd[i]    = IMU_aclm_fltd[i][start_index:end_index]
            IMU_rotation_fltd[i]= IMU_rotation_fltd[i].iloc[start_index:end_index]
            
            
            #---------Equalizing the length of SD1 and SD2 data sets (old data only) --------------
            
            if data_format == '1':
                min_length_sensor = min(len(s3_fltd[i]), len(s4_fltd[i]))
                # min_length for sensor and sensation are equal except for the
                # case of data sets taken with DAQ version 1.0
            
                sensation_data_trimd[i] = sensation_data_trimd[i][:min_length_sensor]
                
                s1_fltd[i]  = s1_fltd[i][:min_length_sensor]
                s2_fltd[i]  = s2_fltd[i][:min_length_sensor]
                s3_fltd[i] = s3_fltd[i][:min_length_sensor]
                s4_fltd[i] = s4_fltd[i][:min_length_sensor]
                s5_fltd[i] = s5_fltd[i][:min_length_sensor]
                s6_fltd[i] = s6_fltd[i][:min_length_sensor]
                Flexi_data_fltd[i]  = Flexi_data_fltd[i][:min_length_sensor]
                IMU_aclm_fltd[i]    = IMU_aclm_fltd[i][:min_length_sensor]
            elif data_format == '2':
                pass
                #Equalization isn't required for new belt data
            
            
            # Update the progress bar
            pbar.update(1)
            
            
    return s1_fltd, s2_fltd, s3_fltd, s4_fltd, s5_fltd, s6_fltd, Flexi_data_fltd, IMU_aclm_fltd, IMU_rotation_fltd, sensation_data_trimd



def get_info_from_preprocessed_data(n_data_files, Flexi_data_fltd, Fs_sensor):
    # Extracting force sensor data during each recording session

    Force_mean = []
    Force_signal_power = []
    # sample_size      = 30; % sample size in seconds
    
    for i in range(n_data_files):
        # Force_data_sample[i] = Flexi_data_fltd[i](:sample_size*Fs_sensor)
        
        Force_data_sample = np.abs(Flexi_data_fltd[i])
        Force_mean.append( np.mean(Force_data_sample) )
        Force_signal_power.append( np.sum(Force_data_sample ** 2) / len(Force_data_sample) )

    # Duration of each data sample after trimming
    duration_trimmed_data_files = []
    for i in range(n_data_files):
        duration_trimmed_data_files.append ( len(Flexi_data_fltd[i]) / (Fs_sensor * 60) )  # Duration in minutes

    return duration_trimmed_data_files, Force_mean


#==============Data and feature extraction part

def get_IMU_map(IMU_data, data_file_name, Fs_sensor, IMU_threshold_, IMU_dilation_, data_format):
    
    """
    Algorithm: @ Abhishek Kumar Ghosh
    Author: @ Moniruzzaman Akash
    Segments and dilates the IMU data and returns the resultant data as IMU map.
    Parameters
    ----------
    IMU_data : List. Shape: (number of files, sensor data values)
        Filtered Sensor data to determine maternal movement.
    data_file_name : str
        A string containing data file name. i.e. 'S1_Day3_dataFile_000.mat'
    Fs_sensor : int
        Sampling rate of the sensor data.
    data_format : str
        Belt selection number. "1"(old), "2"(new)

    Returns
    -------
    IMU_map : numpy vector of boolean values
        Contains True/False values for that particular data file considering IMU Mapping.

    """


    if data_format == '1':
        IMU_threshold = [0.003, 0.002]  # Fixed threshold values obtained through separate testing
        IMU_dilation_time = 4.0
        
        # -----------Segmentation of IMU data and creation of IMU_map--------------
        if getParticipantID(data_file_name) == '1' or getParticipantID(data_file_name) == '2':
            IMU_map = np.abs(IMU_data) >= IMU_threshold[0]  # Threshold for Subject 1 or 2
        else:
            IMU_map = np.abs(IMU_data) >= IMU_threshold[1]  # Threshold for Subject 3, 4, 5
            
            
    elif data_format == '2':
        IMU_threshold = IMU_threshold_
        IMU_dilation_time = IMU_dilation_
        
        # -----------Segmentation of IMU data and creation of IMU_map--------------
        IMU_map = np.abs(IMU_data) >= IMU_threshold

        
    # Dilation length in seconds
    IMU_dilation_size = round(IMU_dilation_time * Fs_sensor)  # Dilation length in sample number

    # ----------------------Dilation of IMU data-------------------------------
    IMU_map = custom_binary_dilation( IMU_map, IMU_dilation_size)# Dilate or expand the ROI's(points with value = 1) by dilation_size (half above and half below),as defined by SE

    return IMU_map

def get_IMU_rot_map(IMU_rotation_data, IMU_rot_threshold, IMU_dilation_time, Fs_sensor):
    
    """
    @ Algorithm and Author: Ahsan Imran, Moniruzzaman Akash
    Segments and dilates the IMU data and returns the resultant data as IMU map.
    Parameters
    ----------
    IMU_data : List. Shape: (number of files, sensor data values)
        Filtered Sensor data to determine maternal movement.
    data_file_name : str
        A string containing data file name. i.e. 'S1_Day3_dataFile_000.mat'
    Fs_sensor : int
        Sampling rate of the sensor data.
    data_format : str
        Belt selection number. "1"(old), "2"(new)

    Returns
    -------
    IMU_map : numpy vector of boolean values
        Contains True/False values for that particular data file considering IMU Mapping.

    """
    
    # find differerntial of each euler angle
    IMU_rotR_fltd = IMU_rotation_data['roll'].values
    IMU_rotP_fltd = IMU_rotation_data['pitch'].values
    IMU_rotY_fltd = IMU_rotation_data['yaw'].values

    # IMU_rotR_diff = np.abs(np.diff(IMU_rotR_fltd))
    # IMU_rotP_diff = np.abs(np.diff(IMU_rotP_fltd))
    # IMU_rotY_diff = np.abs(np.diff(IMU_rotY_fltd))
    
    
    # -----------Segmentation of IMU data and creation of IMU_map--------------
    # threshold for each euler angle and make them true if pass threshold
    IMU_rotR_map = np.abs(IMU_rotR_fltd) >= IMU_rot_threshold
    IMU_rotP_map = np.abs(IMU_rotP_fltd) >= IMU_rot_threshold
    IMU_rotY_map = np.abs(IMU_rotY_fltd) >= IMU_rot_threshold

    # Dilation length in seconds
    IMU_dilation_size = round(IMU_dilation_time * Fs_sensor)# Dilation length in sample number
    
    
    # ----------------------Dilation of IMU data-------------------------------
    IMU_rotR_map = custom_binary_dilation(IMU_rotR_map, IMU_dilation_size)
    IMU_rotP_map = custom_binary_dilation(IMU_rotP_map, IMU_dilation_size)
    IMU_rotY_map = custom_binary_dilation(IMU_rotY_map, IMU_dilation_size)
    
    
    
    IMU_rotation_map = IMU_rotR_map | IMU_rotP_map | IMU_rotY_map
    
    # IMU_rotation_map = np.insert(IMU_rotation_map, 0, 0)

    return IMU_rotation_map

def get_merged_map(IMU_aclm_map, IMU_rot_map):
    """
    Algorithm and AUthor: @ Moniruzzaman Akash
    Generates sensation map by merging IMU_aclm and IMU_rotation map
    Parameters
    ----------
    IMU_aclm_map : numpy.ndarray.
    IMU_rot_map : numpy.ndarray. 
    

    Returns
    -------
    map_final : numpy.ndarray. Shape(sensation mapped values)
        merged map of IMU accelerometer and rotation map
    """
    
    map_added = IMU_aclm_map.astype(int) + IMU_rot_map.astype(int)
    
    map_added[0] = 0
    map_added[-1] = 0
    
    # Find where changes from non-zero to zero or zero to non-zero occur
    changes = np.where((map_added[:-1] == 0) != (map_added[1:] == 0))  [0] + 1


    # Create tuples of every two values
    windows = []
    for i in range(0, len(changes), 2):
        if i < len(changes) - 1:
            windows.append((changes[i], changes[i+1]))
        else:
            windows.append((changes[i],))
    
    map_final = np.zeros(len(map_added))
    
    for window in windows:
        start = window[0]
        
        if len(window) == 2:
            end = window[1]
            if np.any(map_added[start:end] == 2):
                map_final[start:end] = 1
        else:
            map_final[start:] = 1
            
            
    # print("Resulting tuples:", windows)
    return map_final.astype(dtype=bool)

def get_IMU_rot_map_complex(IMU_rotation_data, IMU_rot_threshold, IMU_dilation_time, Fs_sensor):
    """
    Modified function to segment and dilate the IMU data based on new conditions.
    Parameters
    ----------
    IMU_rotation_data : DataFrame. Contains 'roll', 'pitch', 'yaw' columns with values.
    IMU_rot_threshold : int. Not used in the modified function but kept for compatibility.
    IMU_dilation_time : float. Time in seconds for dilation.
    Fs_sensor : int. Sampling rate of the sensor data.

    Returns
    -------
    IMU_rotation_map : numpy array of boolean values. Contains True/False values based on new conditions.
    """
   
    # Extract roll, pitch, and yaw values
    IMU_rotR_fltd = IMU_rotation_data['roll'].values
    IMU_rotP_fltd = IMU_rotation_data['pitch'].values
    IMU_rotY_fltd = IMU_rotation_data['yaw'].values

    # Take the absolute values of the Euler angles
    IMU_rotR_fltd = np.abs(IMU_rotR_fltd)
    IMU_rotP_fltd = np.abs(IMU_rotP_fltd)
    IMU_rotY_fltd = np.abs(IMU_rotY_fltd)
   
    # Initialize the IMU map with zeros (False)
    IMU_rotation_map = np.zeros_like(IMU_rotP_fltd, dtype=bool)
   
    # Apply new conditions to modify the IMU map
    for i in range(len(IMU_rotP_fltd)):
        if IMU_rotP_fltd[i] < 1.5:
            if IMU_rotR_fltd[i] < 4 and IMU_rotY_fltd[i] < 4:
                IMU_rotation_map[i] = False  
            else:
                IMU_rotation_map[i] = True
        else:  # IMU_rotP_fltd[i] >= 1
            if IMU_rotR_fltd[i] > 4 or IMU_rotY_fltd[i] > 4:
                IMU_rotation_map[i] = True  

    # Dilation length in samples
    IMU_dilation_size = round(IMU_dilation_time * Fs_sensor)
   
    # Dilation of IMU data
    IMU_rotation_map = custom_binary_dilation(IMU_rotation_map, IMU_dilation_size)
   
    return IMU_rotation_map

def get_IMU_rot_map_TWO(IMU_rotation_data, IMU_rot_threshold, IMU_dilation_time, Fs_sensor):
    
    """
    @ 
    Segments and dilates the IMU data and returns the resultant data as IMU map.
    Parameters
    ----------
    IMU_data : List. Shape: (number of files, sensor data values)
        Filtered Sensor data to determine maternal movement.
    data_file_name : str
        A string containing data file name. i.e. 'S1_Day3_dataFile_000.mat'
    Fs_sensor : int
        Sampling rate of the sensor data.
    data_format : str
        Belt selection number. "1"(old), "2"(new)

    Returns
    -------
    IMU_map : numpy vector of boolean values
        Contains True/False values for that particular data file considering IMU Mapping.

    """
    
    # find differerntial of each euler angle
    IMU_rotR_fltd = IMU_rotation_data['roll'].values
    IMU_rotP_fltd = IMU_rotation_data['pitch'].values
    IMU_rotY_fltd = IMU_rotation_data['yaw'].values
    
    
    IMU_rotR_fltd = np.abs(IMU_rotR_fltd)
    IMU_rotP_fltd = np.abs(IMU_rotP_fltd)
    IMU_rotY_fltd = np.abs(IMU_rotY_fltd)
    
    
    IMU_rotR_map = IMU_rotR_fltd >= 4.0
    IMU_rotP_map = IMU_rotP_fltd >= 1.5
    IMU_rotY_map = IMU_rotY_fltd >= 4.0

    # Dilation length in seconds
    IMU_dilation_size = round(IMU_dilation_time * Fs_sensor)# Dilation length in sample number
    
    
    IMU_rotation_map_1 = get_merged_map(IMU_rotR_map, IMU_rotP_map)
    IMU_rotation_map_2 = get_merged_map(IMU_rotR_map, IMU_rotY_map)
    IMU_rotation_map_3 = get_merged_map(IMU_rotP_map, IMU_rotY_map)

    IMU_rotation_map = IMU_rotation_map_1 | IMU_rotation_map_2 | IMU_rotation_map_3

    IMU_rotation_map = custom_binary_dilation(IMU_rotation_map, IMU_dilation_size)
    # IMU_rotation_map = IMU_rotR_map | IMU_rotP_map | IMU_rotY_map
    
    # IMU_rotation_map = np.insert(IMU_rotation_map, 0, 0)

    return IMU_rotation_map

def remove_IMU_from_segmentation(IMU_map, segmentation_map):
    """
    Algorithm and AUthor: @ Moniruzzaman Akash
    """
    map_added = IMU_map.astype(int) + segmentation_map.astype(int)
    
    map_added[0] = 0
    map_added[-1] = 0
    
    # Find where changes from non-zero to zero or zero to non-zero occur
    changes = np.where((map_added[:-1] == 0) != (map_added[1:] == 0))  [0] + 1


    # Create tuples of every two values
    windows = []
    for i in range(0, len(changes), 2):
        if i < len(changes) - 1:
            windows.append((changes[i], changes[i+1]))
        else:
            windows.append((changes[i],))
    
    map_final = np.copy(segmentation_map)
    
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
    # print("Resulting tuples:", windows)
    return map_final.astype(dtype=int)

def get_sensation_map(sensation_data, IMU_map, ext_backward, ext_forward, Fs_sensor, Fs_sensation):
    
    """
    Algorithm: @ Abhishek Kumar Ghosh
    Author: @ Moniruzzaman Akash
    Generates sensation map by dilating every detection to past and future. Revomes the windows that overlaps with IMU_map
    Parameters
    ----------
    sensation_data : numpy.ndarray. Shape(sensation data values)
        Sensation data for a particular data file.
    IMU_map : numpy.ndarray. Shape(IMU mapped values)
        IMU Map of a particular data file.
    ext_backward : int
        Backward extension length in second. i.e, 5 seconds. so value = 5
    ext_forward : int
        Forward extension length in second. i.e, 2 seconds. so value = 2
    Fs_sensor : int
        The sampling rate of the sensor data. Usual case scenario Fs_sensor = 1024
    Fs_sensation : int
        The sampling rate of the sensation data. Usual case scenario Fs_sensation = 1024

    Returns
    -------
    M_sntn_map : numpy.ndarray. Shape(sensation mapped values)
        Contains 1/0 values for that particular data file considering Sensation Data Mapping.
        A vector with all the sensation windows joined together
    """

    M_event_index = np.where(sensation_data)[0]  # Sample numbers for maternal sensation detection
    M_sntn_map = np.zeros(len(IMU_map))  # Initializing the map with zeros everywhere

    # Parameters for creating M_sensation_map
    DLB = round(ext_backward * Fs_sensor)  # Backward extension length in number of samples
    DLF = round(ext_forward * Fs_sensor)  # Forward extension length in number of samples

    for j in range(len(M_event_index)):  # M_event_index contains index of all the maternal sensation detections
        # Getting the index values for the map
        L = M_event_index[j]  # Sample no. corresponding to a maternal sensation
        L1 = L * round(Fs_sensor / Fs_sensation) - DLB  # Sample no. for the starting point of this sensation in the map
        L2 = L * round(Fs_sensor / Fs_sensation) + DLF  # Sample no. for the ending point of this sensation in the map
        L1 = max(L1, 0)  # Just a check so that L1 remains higher than or equal to 0
        L2 = min(L2, len(M_sntn_map))  # Just a check so that L2 remains lower than or equal to the last data sample

        # Generating the map - a single vector with all the sensation data mapping
        M_sntn_map[L1:L2 + 1] = 1  # Assigns 1 to the locations defined by L1:L2

        # Removal of the maternal sensation that has coincided with the body movement
        X = np.sum(M_sntn_map[L1:L2 + 1] * IMU_map[L1:L2 + 1])  # This is non-zero if there is a coincidence
        if X:
            M_sntn_map[L1:L2 + 1] = 0  # Removes the sensation data from the map

    return M_sntn_map


def get_segmented_data(sensor_data, min_SN, IMU_map, dilation_time, Fs_sensor):
    """
    Algorithm: @ Abhishek Kumar Ghosh
    Author: @ Moniruzzaman Akash
    Thresholds the data, removes body movement, and dilates the data.
    Parameters
    ----------
    sensor_data : List. Shape(number of FM sensors, sensor data values)
        Sensor data for a particular data file.
    min_SN : List (number of FM sensors)
        These values are selected to get SEN of 99%
    IMU_map : numpy.ndarray. Shape(IMU mapped values)
        IMU Map of a particular data file.
    dilation_time : int
        Dilation size in seconds
    Fs_sensor : int
        The sampling rate of the sensor data. Usual case scenario Fs_sensor = 1024

    Returns
    -------
    sensor_data_sgmntd : List. Shape(number of FM sensors, sensor data values)
        Segmented data values for that particular sensor in that particular datafile.
    h: List. Shape (number of FM sensors)
        Generates a threshold variable based on which TPD,FPD and features can be extracted.

    """
    
    low_signal_quantile = 0.25
    sgmntd_signal_cutoff = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
    dilation_size = round(dilation_time * Fs_sensor)
    SE = np.ones((dilation_size+1))  # linear element necessary for dilation operation

    n_sensors = len(sensor_data)

    h = np.zeros(n_sensors)  # Variable for threshold
    sensor_data_sgmntd = [None] * n_sensors
    
    def getBinaryDialation(index):
        # Determining the threshold
        s = np.abs(sensor_data[index])
        LQ = np.quantile(s, low_signal_quantile)  # Returns the quantile value for low 25% (= low_signal_quantile) of the signal
        e = s[s <= LQ]  # Signal noise
        h[index] = min_SN[index] * np.median(e)  # Threshold value. Each row will contain the threshold value for each data file

        if np.isnan(h[index]):  # Check if h = NaN. This happens when e=[], as median(e)= NaN for that case!!!
            h[index] = np.inf
        if h[index] < sgmntd_signal_cutoff[index]:
            h[index] = np.inf  # Precaution against too noisy signal

        # Thresholding
        each_sensor_data_sgmntd = (s >= h[index]).astype(int)  # Considering the signals that are above the threshold value; data point for noise signal becomes zero and signal becomes 1

        # Exclusion of body movement data
        each_sensor_data_sgmntd *= (1 - IMU_map)  # Exclusion of body movement data

        # Dilation of the thresholded data
        # print("\t\tDialating sensor data {}".format(index))
        #sensor_data_sgmntd.append( binary_dilation(each_sensor_data_sgmntd, structure=SE) )# For individual sensor performance
        
        
        # return  (binary_dilation(each_sensor_data_sgmntd, structure=SE), index)
        return  (custom_binary_dilation(each_sensor_data_sgmntd, dilation_size+1), index)
    
    # Thresholding
    
    # Create a ThreadPoolExecutor with a specified number of threads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the tasks to the executor and store the Future objects
        futures = [executor.submit(getBinaryDialation, j) for j in range(n_sensors)]
    
        # Retrieve the results as they become available
        result = [future.result() for future in concurrent.futures.as_completed(futures)]
        
    for r in result:
        index = r[1]#index of the result
        sensor_data_sgmntd[index] = r[0]


    return sensor_data_sgmntd, h

def get_segmented_data_cmbd_all_sensors(s1_fltd, s2_fltd, s3_fltd, s4_fltd, s5_fltd, s6_fltd,
                                        FM_min_SN, IMU_map_, FM_dilation_time, Fs_sensor, sensor_selection):
    if sensor_selection == 1:
       n_FM_sensors = 2
       sensor_data_fltd = [s1_fltd, s2_fltd]
       sensor_data_sgmntd, threshold_ = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map_, FM_dilation_time, Fs_sensor)
       for i in range(n_FM_sensors):
           sensor_data_sgmntd[i] = remove_IMU_from_segmentation(IMU_map_, sensor_data_sgmntd[i])
       sensor_data_sgmntd_cmbd_all_sensors = sensor_data_sgmntd[0] | sensor_data_sgmntd[1]
    elif sensor_selection == 2:
        n_FM_sensors = 2
        sensor_data_fltd = [s3_fltd, s4_fltd]
        sensor_data_sgmntd, threshold_ = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map_, FM_dilation_time, Fs_sensor)
        for i in range(n_FM_sensors):
            sensor_data_sgmntd[i] = remove_IMU_from_segmentation(IMU_map_, sensor_data_sgmntd[i])
        sensor_data_sgmntd_cmbd_all_sensors = sensor_data_sgmntd[0] | sensor_data_sgmntd[1]
    elif sensor_selection == 3:
        n_FM_sensors = 2
        sensor_data_fltd = [s5_fltd, s6_fltd]
        sensor_data_sgmntd, threshold_ = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map_, FM_dilation_time, Fs_sensor)
        for i in range(n_FM_sensors):
            sensor_data_sgmntd[i] = remove_IMU_from_segmentation(IMU_map_, sensor_data_sgmntd[i])
        sensor_data_sgmntd_cmbd_all_sensors = sensor_data_sgmntd[0] | sensor_data_sgmntd[1]
    elif sensor_selection == 4:
        n_FM_sensors = 4
        sensor_data_fltd = [s1_fltd, s2_fltd, s3_fltd, s4_fltd]
        sensor_data_sgmntd, threshold_ = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map_, FM_dilation_time, Fs_sensor)
        for i in range(n_FM_sensors):
            sensor_data_sgmntd[i] = remove_IMU_from_segmentation(IMU_map_, sensor_data_sgmntd[i])
        sensor_data_sgmntd_cmbd_all_sensors = sensor_data_sgmntd[0] | sensor_data_sgmntd[1] | sensor_data_sgmntd[2] | sensor_data_sgmntd[3]
    elif sensor_selection == 5:
        n_FM_sensors = 4
        sensor_data_fltd = [s1_fltd, s2_fltd, s5_fltd, s6_fltd]
        sensor_data_sgmntd, threshold_ = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map_, FM_dilation_time, Fs_sensor)
        for i in range(n_FM_sensors):
            sensor_data_sgmntd[i] = remove_IMU_from_segmentation(IMU_map_, sensor_data_sgmntd[i])
        sensor_data_sgmntd_cmbd_all_sensors = sensor_data_sgmntd[0] | sensor_data_sgmntd[1] | sensor_data_sgmntd[2] | sensor_data_sgmntd[3]
    
    elif sensor_selection == 6:
        n_FM_sensors = 4
        sensor_data_fltd = [s3_fltd, s4_fltd, s5_fltd, s6_fltd]
        sensor_data_sgmntd, threshold_ = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map_, FM_dilation_time, Fs_sensor)
        for i in range(n_FM_sensors):
            sensor_data_sgmntd[i] = remove_IMU_from_segmentation(IMU_map_, sensor_data_sgmntd[i])
        sensor_data_sgmntd_cmbd_all_sensors = sensor_data_sgmntd[0] | sensor_data_sgmntd[1] | sensor_data_sgmntd[2] | sensor_data_sgmntd[3]
        
    elif sensor_selection == 7:
        n_FM_sensors = 6
        sensor_data_fltd = [s1_fltd, s2_fltd, s3_fltd, s4_fltd, s5_fltd, s6_fltd]
        sensor_data_sgmntd, threshold_ = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map_, FM_dilation_time, Fs_sensor)
        for i in range(n_FM_sensors):
            sensor_data_sgmntd[i] = remove_IMU_from_segmentation(IMU_map_, sensor_data_sgmntd[i])
        # sensor_data_sgmntd_all.append(sensor_data_sgmntd)
        
        sensor_data_sgmntd_cmbd_all_sensors = sensor_data_sgmntd[0] | sensor_data_sgmntd[1] | sensor_data_sgmntd[2] | sensor_data_sgmntd[3] | sensor_data_sgmntd[4] | sensor_data_sgmntd[5]

    return sensor_data_sgmntd_cmbd_all_sensors, sensor_data_fltd, sensor_data_sgmntd, n_FM_sensors, threshold_




def match_with_m_sensation(sensor_data_sgmntd, sensation_data, IMU_map, M_sntn_Map,
                           ext_backward, ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation):
    
    # % MATCH_WITH_M_SENSATION Summary of this function goes here
    # %   Input variables:  sensor_data_sgmntd- a cell variable
    # %                                         Each cell contains data from a sensor or a combination.
    # %                     sensation_data, IMU_map, M_sntn_Map- cell variables with single cell   
    # %                     ext_bakward, ext_forward, FM_dilation_time- scalar values 
    # %                     Fs_sensor, Fs_sensation- scalar values
    # %   Output variables: TPD, FPD, TND, FND- vectors with number of rows equal to the 
    # %                                         number of cells in the sensor_data_sgmntd

    # Calculation of fixed values
    n_sensors = len(sensor_data_sgmntd)
    matching_window_size = ext_backward + ext_forward  # window size is equal to the window size used to create the maternal sensation map
    minm_overlap_time = FM_dilation_time / 2  # Minimum overlap in second
    DLB = round(ext_backward * Fs_sensor)  # backward extension length in sample number
    DLF = round(ext_forward * Fs_sensor)  # forward extension length in sample number

    # Variable declaration
    TPD = np.zeros(n_sensors)  # True positive detection
    FND = np.zeros(n_sensors)  # False negative detection
    TND = np.zeros(n_sensors)  # True negative detection
    FPD = np.zeros(n_sensors)  # False positive detection

    # Labeling sensation data and determining the number of maternal sensation detection
    sensation_data_labeled = label(sensation_data)
    n_Maternal_detected_movement = len(np.unique(sensation_data_labeled)) - 1

    # Loop for matching individual sensors
    for j in range(n_sensors):
        # ------------------ Determination of TPD and FND ----------------%    
        if n_Maternal_detected_movement:  # When there is a detection by the mother
            for k in range(1, n_Maternal_detected_movement + 1):
                L_min = np.where(sensation_data_labeled == k)[0][0]  # Sample no. corresponding to the start of the label
                L_max = np.where(sensation_data_labeled == k)[0][-1]  # Sample no. corresponding to the end of the label

                L1 = L_min * round(Fs_sensor / Fs_sensation) - DLB  # sample no. for the starting point of this sensation in the map
                L1 = max(L1, 0)  # Just a check so that L1 remains higher than 1st data sample

                L2 = L_max * round(Fs_sensor / Fs_sensation) + DLF  # sample no. for the ending point of this sensation in the map
                L2 = min(L2, len(sensor_data_sgmntd[j]))  # Just a check so that L2 remains lower than the last data sample

                indv_sensation_map = np.zeros(len(sensor_data_sgmntd[j]))  # Need to be initialized before every detection matching
                indv_sensation_map[L1:L2] = 1  # mapping individual sensation data

                X = np.sum(indv_sensation_map * IMU_map)  # this is non-zero if there is a coincidence with maternal body movement

                if not X:  # true when there is no coincidence
                    # TPD and FND calculation for individual sensors
                    Y = np.sum(sensor_data_sgmntd[j] * indv_sensation_map)  # Non-zero value gives the matching
                    if Y:  # true if there is a coincidence
                        TPD[j] += 1  # TPD incremented
                    else:
                        FND[j] += 1  # FND incremented

        # ------------------- Determination of TND and FPD  ------------------%    
        # Removal of the TPD and FND parts from the individual sensor data
        sensor_data_sgmntd_labeled = label(sensor_data_sgmntd[j])
        curnt_matched_vector = sensor_data_sgmntd_labeled * M_sntn_Map  # Non-zero elements give the matching. In M-sntn_Map multiple windows can overlap, which was not the case in the sensation_data
        curnt_matched_label = np.unique(curnt_matched_vector)  # Gives the label of the matched sensor data segments
        Arb_val = 4  # An arbitrary value
        
        
        if len(curnt_matched_label) > 1:
            curnt_matched_label = curnt_matched_label[1:]  # Removes the first element, which is 0
            for m in range(len(curnt_matched_label)):
                sensor_data_sgmntd[j][sensor_data_sgmntd_labeled == curnt_matched_label[m] ] = Arb_val
                # Assigns an arbitrary value to the TPD segments of the segmented signal

        sensor_data_sgmntd[j][M_sntn_Map == 1] = Arb_val  # Assigns an arbitrary value to the area under the M_sntn_Map
        sensor_data_sgmntd_removed = sensor_data_sgmntd[j][sensor_data_sgmntd[j] != Arb_val]  # Removes all the elements with value = Arb_val from the segmented data

        # Calculation of TND and FPD for individual sensors
        L_removed = len(sensor_data_sgmntd_removed)
        index_window_start = 0
        index_window_end = int( min(index_window_start + Fs_sensor * matching_window_size, L_removed) )
        while index_window_start < L_removed:
            indv_window = sensor_data_sgmntd_removed[index_window_start: index_window_end]
            index_non_zero = np.where(indv_window)[0]

            if len(index_non_zero) >= (minm_overlap_time * Fs_sensor):
                FPD[j] += 1
            else:
                TND[j] += 1

            index_window_start = index_window_end + 1
            index_window_end = int( min(index_window_start + Fs_sensor * matching_window_size, L_removed) )

    return TPD, FPD, TND, FND


def get_performance_params(TPD_all, FPD_all, TND_all, FND_all):
    """
    Algorithm: @ Abhishek Kumar Ghosh
    Author: @ Moniruzzaman Akash
    Gets the performance values of the implemented algorithms
    Parameters
    ----------
    TPD_all : single cell/multi-cell variable
        Number of cells indicates the number of sensor data or combination data provided together.
    FPD_all : single cell/multi-cell variable
        Number of cells indicates the number of sensor data or combination data provided together..
    TND_all : single cell/multi-cell variable
        Number of cells indicates the number of sensor data or combination data provided together..
    FND_all : single cell/multi-cell variable
        Number of cells indicates the number of sensor data or combination data provided together..
        
    Each cell contains a vector with row size = n_data_files.
    Returns
    -------
    SEN_all : cell variable with size same as the input variables.
        DESCRIPTION.
    PPV_all : cell variable with size same as the input variables.
        DESCRIPTION.
    SPE_all : cell variable with size same as the input variables.
        DESCRIPTION.
    ACC_all : cell variable with size same as the input variables.
        DESCRIPTION.
    FS_all : cell variable with size same as the input variables.
        DESCRIPTION.
    FPR_all : cell variable with size same as the input variables.
        DESCRIPTION.

    """

    n_sensors = len(TPD_all)

    SEN_all = np.zeros(n_sensors)
    PPV_all = np.zeros(n_sensors)
    SPE_all = np.zeros(n_sensors)
    ACC_all = np.zeros(n_sensors)
    FS_all  = np.zeros(n_sensors)
    FPR_all = np.zeros(n_sensors)

    B = 1  # Beta value for F_B score calculation.

    for i in range(n_sensors):
        SEN_all[i] = TPD_all[i] / (TPD_all[i] + FND_all[i])  # Operates on individual data sets
        PPV_all[i] = TPD_all[i] / (TPD_all[i] + FPD_all[i])
        SPE_all[i] = TND_all[i] / (TND_all[i] + FPD_all[i])
        ACC_all[i] = (TPD_all[i] + TND_all[i]) / (TPD_all[i] + FND_all[i] + FPD_all[i] + TND_all[i])
        FS_all[i] = (1 + B ** 2) * (PPV_all[i] * SEN_all[i]) / (B ** 2 * PPV_all[i] + SEN_all[i])

        FPR_all[i] = FPD_all[i] / (FPD_all[i] + TND_all[i])

    return SEN_all, PPV_all, SPE_all, ACC_all, FS_all, FPR_all
