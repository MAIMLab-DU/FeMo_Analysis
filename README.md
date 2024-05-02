## Installation
1. Install Anaconda on your device 
2. Open Anaconda Prompt
3. Create a conda environemnt on Anaconda using following command on prompt:
   ```
   conda create --name FeMo python=3.10.3
   ```
   Note: This command create a conda environemnt named 'new_data' with python version 3.10.3, which is compatible for our code. You can choose your prefered name by replacing 'new_data' with the name you want.
   
4. Activate the conda environemnt that you created:
   ```
   conda activate FeMo
   ```
5. Navigate to the folder where your code is situated:
   ```
   cd /d <folder path>
   ```
      - For example:
         ```
         cd  /d   D:\FeMo_Analysis
         ```
6. Install all necessary package:
   ```
   pip install -r requirement.txt
   ```
7. Run Inference:
   ```
   python FeMo_Analysis.py
   ```