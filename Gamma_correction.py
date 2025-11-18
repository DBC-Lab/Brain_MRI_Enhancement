import os
import glob
import SimpleITK as sitk
import numpy as np

input_dir = "./"  

file_pattern = os.path.join(input_dir, "*-reorient.hdr")

file_list = glob.glob(file_pattern)

if not file_list:
    print(f"No files matching pattern '{file_pattern}' found.")
else:
    print(f"Found {len(file_list)} files matching pattern '{file_pattern}'.")

for file_path in file_list:
    print(f"Processing file: {file_path}")

    img = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(img)  

    gamma = 1.3
    normalized = data / np.max(data) 
    adjusted_data = np.power(normalized, gamma) * np.max(data)  

    adjusted_img = sitk.GetImageFromArray(adjusted_data)
    adjusted_img.CopyInformation(img)  

    output_path = file_path.replace("-reorient", "-reorient-adjusted1p3")
    sitk.WriteImage(adjusted_img, output_path)

    print(f"Saved adjusted image to: {output_path}")
