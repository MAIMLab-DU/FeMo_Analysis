# -*- coding: utf-8 -*-
"""
This class is developed to download and list s3 bucket device data

@author: akash
"""
import boto3, os

class S3FileManager:
    def __init__(self, bucket_name='femo-sensor-logfiles'):
        self.bucket_name = bucket_name
        self.s3 = boto3.client('s3')
        
        
    def download_file(self, object_key):
        current_script_path = os.path.realpath(__file__)
        folder = current_script_path+ "/Data_files/"+object_key.split("/")[0].replace(":", "_") #Convert from 'DC:54:75:C2:E3:FC/' to 'DC_54_75_C2_E3_FC/'
        file_name = object_key.split("/")[1]
        
        if not os.path.exists(folder):
            # Create the directory
            os.makedirs(folder)

        data_file_path = folder+"/"+file_name+".dat"  # Specify the local file path where you want to save the file
        
        try:
            #If file is not downloaded before downlaod it
            if not os.path.exists(data_file_path):
                print(f"Downloading {data_file_path}")
                self.s3.download_file(self.bucket_name, object_key, data_file_path)
            else:
                print(f"File already downlaoded: {file_name}")
                
        except Exception as e:
            print(f"Error downloading file: {e}")

    def download_files_in_device(self, device_name, local_dir):
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=device_name)
            if 'Contents' in response:
                for obj in response['Contents']:
                    obj_key = obj['Key']
                    self.download_file(obj_key)
            else:
                print("No objects found with the specified device.")
        except Exception as e:
            print(f"Error downloading files with key part: {e}")
    
    def get_fileList_in_device(self, device_name):
        #Retrieves a list of files and their corresponding sizes within a specific folder of the S3 bucket.
        files = []
        file_size = []
        paginator = self.s3.get_paginator('list_objects_v2')
        operation_parameters = {'Bucket': self.bucket_name, 'Prefix': device_name}
        page_iterator = paginator.paginate(**operation_parameters)

        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    files.append(obj['Key'])
                    file_size.append(obj['Size'])

        return files, file_size
    
    def get_device_list(self):
        #Return a list of available folders in the bucket
        folders = []
        
        paginator = self.s3.get_paginator('list_objects_v2')
        operation_parameters = {'Bucket': self.bucket_name, 'Delimiter': '/'}
        page_iterator = paginator.paginate(**operation_parameters)

        for page in page_iterator:
            if 'CommonPrefixes' in page:
                for prefix in page['CommonPrefixes']:
                    folders.append(prefix['Prefix'])

        return folders