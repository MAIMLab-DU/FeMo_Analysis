# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:46:49 2024

@author: Monaf
"""
from femo import Femo# Local code imports
import pandas as pd
import numpy as np
from tqdm import tqdm 
import csv
from skimage.measure import label
import glob
import os, time, sys, argparse, warnings, logging
from tqdm import tqdm




def dat_to_csv(data_file_names, output_dir):
    
    # data_file_names = get_data_file_names(input_dir)
    output_file_path_list = []
    number_of_files = len (data_file_names) # Total number of files
    # output_dir = 'D:/Monaf/New Data FM Monitoring_5_participants_to_csv/'
    
    with tqdm(total=number_of_files) as pbar:
        for i in range (number_of_files):
            device_name = data_file_names[i].split('/')[-2] + '/'
            file_name = data_file_names[i].split('/')[-1].split('.')[0]
            
            read_data = Femo(data_file_names[i])
        
            all_sensor_df = (read_data.dataframes["piezos"]
                            .join(read_data.dataframes["accelerometers"])
                            .join(read_data.dataframes["imu"])
                            .join(read_data.dataframes["force"])
                            .join(read_data.dataframes["push_button"])
                            .join(read_data.dataframes["timestamp"]))
            
            # Define the directory where you want to save the CSV file
            output_folder = os.path.join(output_dir,device_name)
            
            # Ensure the directory exists
            os.makedirs(output_folder, exist_ok = True)
            
            # Define the full path for the CSV file
            output_file_path = os.path.join(output_folder, f'{file_name}.csv')
            output_file_path_list.append(output_file_path)
            
            # Save the DataFrame to a CSV file
            all_sensor_df.to_csv(output_file_path, index=False)
            print(f"\nCSV file has been created and saved to {output_file_path}")
            
            # Update the progress bar
            pbar.update(1)
            
    return output_file_path_list
    
def get_data_file_names(folder_path, participant_folders = None):
    
# =============================================================================
#     participant_folders = ["F4_12_FA_8D_05_EC",
#                                 "DC_54_75_C2_23_28",
#                                 "F4_12_FA_8D_12_D4",
#                                 "DC_54_75_C0_E8_30",
#                                 "F4_12_FA_8B_1B_C0",
#                                 "DC_54_75_C2_22_4C"]    
# =============================================================================

    # Initialize an empty list to store file names
    file_list = []
        
    for participant_folder in participant_folders:
        participant_path = os.path.join(folder_path, participant_folder)
        if os.path.isdir(participant_path):
            print(participant_folder, "> Found")
            # Get list of files in the participant folder
            files = os.listdir(participant_path)
            
            # Iterate over each file
            for file_name in files:
                # Check the file extension based on data_format
                if (file_name.endswith(".dat")):
                    # If it matches, append the file path to the list
                    file_list.append(folder_path + participant_folder + "/" + file_name)
        else:
            print(participant_folder, "> Not Found")

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

if __name__ == "__main__":

    input_dir = "D:/Monaf/New Data FM Monitoring_5_participants/"
    output_dir = "D:/Monaf/New Data FM Monitoring_5_participants_to_csv/"   
    data_file_names = list_all_files(input_dir)
    # data_file_names = get_data_file_names(input_dir, participant_data_in_use)
    output_file_path_list = dat_to_csv(data_file_names, output_dir)
    




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
