import subprocess
import argparse
#import re
import os
#import threading

parser = argparse.ArgumentParser(
    description="Brain MRIs Enhancement Foundation Model (BME-X)\n"
                "Version: v1.0.4\n"
                "Authors: Yue Sun, Limei Wang, Gang Li, Weili Lin, Li Wang\n"
                "Reference: A foundation model for enhancing magnetic resonance images and downstream "
                "segmentation, registration, and diagnostic tasks, Nat. Biomed. Eng 9, 521–538 (2025).https://doi.org/10.1038/s41551-024-01283-7\n"
                "Contacts: li_wang@med.unc.edu, yuesun@med.unc.edu\n"
                "Code: <https://github.com/DBC-Lab/Brain_MRI_Enhancement>\n"
                "------------------------------------------",
    formatter_class=argparse.RawTextHelpFormatter
)
        
parser.add_argument("--bids_dir", type=str, required=True, help="BIDS dataset directory")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g., sub-0001)")
parser.add_argument("--session", type=str, required=True, help="Session ID (e.g., ses-V01)")
parser.add_argument("--suffix", type=str, required=True, help="Suffix (e.g., T1w)")
    
args = parser.parse_args()
suffix=args.suffix

# Function to ensure the necessary directories exist
def ensure_path_exists(path, mode=0o755):
    """
    Ensures the specified path exists. If it doesn't, create it and set permissions.

    Args:
        path (str): The path to ensure exists.
        mode (int): The permissions to set for the created directories (default is 0o755).
    """
    if not os.path.exists(path):
        print(f"{path} does not exist. Creating it!")
        os.makedirs(path, mode=mode)
        print(f"Directory {path} created with permissions {oct(mode)}.")
    else:
        print(f"{path} already exists.")

# Function to save output in the proper session folder
def save_output(output_dir, subject, session):
    # Create necessary session directories
    session_output_dir = os.path.join(output_dir, f"{subject}", f"{session}", "anat")
    ensure_path_exists(session_output_dir)
    
save_output(args.output_dir, args.subject, args.session)    
    
result1 = subprocess.run([
    'python3', 'BIDS_data.py',
    '--bids_dir', args.bids_dir,
    '--output_dir', args.output_dir,
    '--subject', args.subject,
    '--session', args.session
], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

output = result1.stdout

for line in output.splitlines():
    print(line)
    
age_output = [line for line in output.splitlines() if "Age in months:" in line]
#input_output = [line for line in output.splitlines() if "Paths to test images:" in line]

for line in age_output:
    print(line)

age_str = age_output

age_cleaned= age_str[0].replace('Age in months: ', '')
age_in_month = int(age_cleaned)
if age_in_month>=21:
    age_in_month = '24'
elif 15 <= age_in_month < 21:
    age_in_month = '18'
elif 10 <= age_in_month < 15:
    age_in_month = '12'
elif 7 <= age_in_month < 10:
    age_in_month = '9'
elif 5 <= age_in_month < 7:
    age_in_month = '6'
elif 2 <= age_in_month < 5:
    age_in_month = '3'
elif -0.00001 <= age_in_month < 2:
    age_in_month = '0'
elif age_in_month < 0:
    age_in_month = 'fetal'
else: 
    print('No age information, please add it.')
    
#for line in input_output:
#    print(line)
  
input_path=os.path.join(args.bids_dir, f"{args.subject}", f"{args.session}", "anat")    
output_path=os.path.join(args.output_dir, f"{args.subject}", f"{args.session}", "anat")
print('input_path:', input_path)
print('output_path:', output_path)

import subprocess

try:
    result2 = subprocess.Popen([
        'python3', 'BME_X_enhanced.py',
        '--input_path', input_path,
        '--output_path', output_path,
        '--age_in_month', age_in_month,
        '--suffix', suffix
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    for line in result2.stdout:
        print(line, end='') 
        
    result2.wait()

except subprocess.CalledProcessError as e:
    print(f"Error executing script: {e}")
    print(f"Standard error output:\n{e.stderr}")
    raise
    
# result2 = subprocess.run([
#     'python3', 'BME_X_enhanced.py',
#     '--input_path', input_path,
#     '--output_path', output_path,
#     '--age_in_month', age_in_month,
#     '--suffix', suffix
# ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

# output = result2.stdout

# for line in output.splitlines():
#     print(line)
    
def change_permissions_recursively(directory, mode):
    # Traverse all files and subdirectories under the specified directory
    for root, dirs, files in os.walk(directory):
        # Change the permissions of the current directory
        os.chmod(root, mode)
        
        # Change the permissions of all files
        for file in files:
            file_path = os.path.join(root, file)
            os.chmod(file_path, mode)
            
# Set the target directory path and permission mode (e.g., read, write, execute for all users 0o777)
directory_path = args.output_dir
mode = 0o777  # Read, write, and execute permissions for all users

# Change the permissions of all files and directories under the specified directory
change_permissions_recursively(directory_path, mode)    