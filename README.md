# BME-X: A foundation model for enhancing magnetic resonance images and downstream segmentation, registration and diagnostic tasks
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14047881.svg)](https://doi.org/10.5281/zenodo.14047881)

### A foundation model for the motion correction, super resolution, denoising and harmonization of magnetic resonance images, can improve the performance of machine-learning models across a wide range of tasks.

## Documentation
Our documentation is [here](https://brain-mri-enhancement.readthedocs.io/en/latest/).

## Support
- **Issue**: If you encounter any issues or have concerns, please submit them here https://github.com/DBC-Lab/Brain_MRI_Enhancement/issues

## Contributing
We welcome contributions! Please check out our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started.

## Update (07/29/2025, yuesun814/bme-x:v1.0.4)

Replacing relative paths (e.g., ./...) with absolute paths (e.g., /...) to ensure correct resolution within the container environment.

To pull the image, use the following command:

    docker pull yuesun814/bme-x:v1.0.4
    
## Update (03/23/2025, yuesun814/bme-x:v1.0.3)
We have integrated the [LifespanStrip](https://github.com/DBC-Lab/Atlases-empowered_Lifespan_Skull_Stripping.git) framework and the [BME-X](https://github.com/DBC-Lab/Brain_MRI_Enhancement.git) model into a single Docker image to make it more convenient for everyone to use. By inputting T1w/T2w raw images, this pipeline goes through RAI orientation, intensity inhomogeneity correction, skull stripping, and image enhancement for the brain region. Additionally, the **Quality Index (QI)** of the original images is provided for reference. Please note that the BME-X model in version v1.0.3 is the same as in v1.0.2.  

<div align="center">
<img src="https://github.com/YueSun814/Img-folder/blob/main/LifespanStrip_BME-X1.png" width="100%"> 
</div> 

To pull the image, use the following command:

    docker pull yuesun814/bme-x:v1.0.3

#### If you use yuesun814/bme-x:v1.0.3, please cite the following two papers:

Sun, Y., Wang, L., Li, G. et al. A foundation model for enhancing magnetic resonance images and downstream segmentation, registration and diagnostic tasks. Nat. Biomed. Eng 9, 521–538 (2025). https://doi.org/10.1038/s41551-024-01283-7

Wang, L., Sun, Y., Seidlitz, J. et al. A lifespan-generalizable skull-stripping model for magnetic resonance images that leverages prior knowledge from brain atlases. Nat. Biomed. Eng 9, 700–715 (2025). https://doi.org/10.1038/s41551-024-01337-w

#### How to Run the Container

1. Basic Command
   
    Run the Docker container using the following command:

    ```
    docker run --gpus all -v /path/to/input:/app/data yuesun814/bme-x:v1.0.3 --bids_root filename_of_BIDS_dataset --subject_id id_of_subject --session_id id_of_session
    ```
    ***'-v /path/to/input'*** mounts the input data directory to the container's ***'/app/data'*** directory.
   
    ***'--bids_root'*** specifies the BIDS dataset to be processed.
   
    ***'--subject_id'*** specifies a subject within the BIDS dataset to be processed (optional).
   
    ***'--session_id'*** specifies a session within the BIDS dataset to be processed (optional).
   

2. Example Usage
   
    For example, using the _test_BIDS_ we provided. The following command will process all the data that meets the criteria within the _test_BIDS_raw_.

    ```
    docker run --gpus all -v /home/user/data:/app/data yuesun814/bme-x:v1.0.3 --bids_root test_BIDS_raw
    ```

    The following command will process a specific subject when the ***'--subject_id'*** is provided (e.g. 0001).
    ```
    docker run --gpus all -v /home/user/data:/app/data yuesun814/bme-x:v1.0.3 --bids_root test_BIDS_raw --subject_id 0001
    ```

    The following command will process a specific session when the ***'--session_id'*** (e.g. V02) is provided.
    ```
    docker run --gpus all -v /home/user/data:/app/data yuesun814/bme-x:v1.0.3 --bids_root test_BIDS_raw --session_id V02
3. Help information

    ```
    >>> docker run --gpus all -v /home/user/data:/app/data yuesun814/bme-x:v1.0.3 --help
    
    usage: BME-X.py [-h] [--bids_root BIDS_ROOT] [--subject_id SUBJECT_ID]
                [--session_id SESSION_ID]

    This is for skull stripping (LifespanStrip) and brain MRIs enhancement (BME-X),by Developing Brain Computing Lab, University of North Carolina at Chapel Hill
    Version: v1.0.3
    Authors: Li Wang, Yue Sun, Limei Wang

    If you find it useful, please kindly cite:
    (1) Sun Y, et al. A foundation model for enhancing magnetic resonance images and downstream segmentation, registration, and diagnostic tasks [J]. Nature Biomedical Engineering,     2024: 1-18. https://doi.org/10.1038/s41551-024-01283-7
    (2) Wang L, et al. A lifespan-generalizable skull-stripping model for magnetic resonance images that leverages prior knowledge from brain atlases [J]. Nature Biomedical         Engineering, 2025: 1-16. https://doi.org/10.1038/s41551-024-01337-w

    Contacts:
      - li_wang@med.unc.edu
      - yuesun@med.unc.edu
      - limeiw@med.unc.edu

    Code Repository:
      - https://github.com/DBC-Lab/Brain_MRI_Enhancement
      - https://github.com/DBC-Lab/Atlases-empowered_Lifespan_Skull_Stripping

    Copyright (C) 2025 DBC Lab. All rights reserved.
    ------------------------------------------

    Input:

    optional arguments:
      -h, --help               show this help message and exit
      --bids_root BIDS_ROOT
                            BIDS dataset directory
      --subject_id SUBJECT_ID
                            subject_id
      --session_id SESSION_ID
                            session_id

    Output:

    *-brainmask.nii.gz
                    Brain mask generated by LifespanStrip.
    *-brain-enhanced.nii.gz
                    Brain image enhancement generated by BME-X.
    *-enhanced.nii.gz
                    Enhanced brain image including skull parts.
    *-QI.txt
                    Quality index [0,  ], where a higher value indicates better quality (0 being the worst).
    
## Update (03/16/2025, yuesun814/bme-x:v1.0.2): 
We have updated the BME-X models, trained using the **PyTorch** framework, for enhancing both **T1w** and **T2w** images. The enhancement model was trained with both the **cerebrum** and **cerebellum**, and the **skull part** was reintroduced after enhancement. 
<div align="center">
<img src="https://github.com/YueSun814/Img-folder/blob/main/BME-X_v2.png" width="80%"> 
</div> 

To pull the image, use the following command:

    docker pull yuesun814/bme-x:v1.0.2

#### RAI orientation, Intensity inhomogeneity correction and Brain mask:

We recommend using the [LifespanStrip framework](https://github.com/DBC-Lab/Atlases-empowered_Lifespan_Skull_Stripping.git) to obtain the required inputs for the BME-X model. 
After applying the LifespanStrip framework to raw MR images, please rename the output files as follows:

    *_T?w-reorient-n4.nii.gz → *_T?w.nii.gz

    *_T?w-reorient-brainmask.nii.gz → *_T?w-brainmask.nii.gz

#### a. To run the Docker image with [BIDS files](https://bids.neuroimaging.io/) test data, please use the following command: 

    mkdir -p /Path/to/output && \
    docker run --gpus '"device=GPU_ID"' --user $(id -u):$(id -g) -it --rm \
      -v /Path/to/BIDS/data:/test \
      -v /Path/to/output:/output \
      yuesun814/bme-x:v1.0.2 /usr/bin/python3 /BME_X.py \
      --bids_dir /test \
      --output_dir /output \
      --subject SUBJECT \
      --session SESSION \
      --suffix SUFFIX
      
For example, if the GPU ID is 2, the path to the BIDS data is '/BME_X/test', the path to the output directory is '/BME_X/output', subject is 'sub-0001', session is 'ses-V01' and suffix is 'T2w':

    mkdir -p /BME_X/output && \
    docker run --gpus '"device=2"' --user $(id -u):$(id -g) -it --rm \
      -v /BME_X/test:/test \
      -v /BME_X/output:/output \
      yuesun814/bme-x:v1.0.2 /usr/bin/python3 /BME_X.py \
      --bids_dir /test \
      --output_dir /output \
      --subject sub-0001 \
      --session ses-V01 \
      --suffix T2w
      
You can use the lifespan test data in _test_BIDS_withBrainMask_.

#### b. Directly provide the paths for the input data (e.g., '/BME_X/test/sub-0001/ses-V01/anat') and output results (e.g., '/BME_X/output'), as well as age_in_month and suffix:

    mkdir -p /BME_X/output && \
    docker run --gpus '"device=0"' --user $(id -u):$(id -g) -it --rm \
      -v /BME_X/test:/test \
      -v /BME_X/output:/output \
      yuesun814/bme-x:v1.0.2 /usr/bin/python3 /BME_X_enhanced.py  \
      --input_path /sub-0001/ses-V01/anat \
      --output_path /output/ \
      --age_in_month 72 \
      --suffix T2w      

## Update (12/01/2024, yuesun814/bme-x:v1.0.1): 
We have provided a **Docker** image with all the necessary prerequisites installed for working with the BME-X model and [BIDS files](https://bids.neuroimaging.io/). The recommended CUDA version on your host is V12.2.140.

To pull the image, use the following command:

    docker pull yuesun814/bme-x:v1.0.1

To run the Docker image, please use the following command: 

    mkdir -p /Path/to/output && \
    docker run --gpus '"device=GPU_ID"' --user $(id -u):$(id -g) -it --rm \
      -v /Path/to/BIDS/data:/test \
      -v /Path/to/output:/output \
      yuesun814/bme-x:v1.0.1 /usr/bin/python3 /BME_X.py \
      --bids_dir /test \
      --output_dir /output \
      --subject SUBJECT \
      --session SESSION
      
For example, if the GPU ID is 2, the path to the BIDS data is '/BME_X/test', the path to the output directory is '/BME_X/output', subject is 'sub-0001', and session is 'ses-V01':

    mkdir -p /BME_X/output && \
    docker run --gpus '"device=2"' --user $(id -u):$(id -g) -it --rm \
      -v /BME_X/test:/test \
      -v /BME_X/output:/output \
      yuesun814/bme-x:v1.0.1 /usr/bin/python3 /BME_X.py \
      --bids_dir /test \
      --output_dir /output \
      --subject sub-0001 \
      --session ses-V01 
You can use the lifespan test data in _test_BIDS_.

----------------------------------------------------------------------------------------

<div align="center">
<img src="https://github.com/YueSun814/Img-folder/blob/main/BME-X_intro.png" width="100%"> 
</div> 

## Version
Current version: 1.0.1 ([DOI: 10.5281/zenodo.14047881](https://doi.org/10.5281/zenodo.14047881))

## Papers
Sun, Y., Wang, L., Li, G. et al. A foundation model for enhancing magnetic resonance images and downstream segmentation, registration and diagnostic tasks. Nat. Biomed. Eng (2024). https://doi.org/10.1038/s41551-024-01283-7

## Intended Usage
BME-X is designed for researchers and clinicians working with structural MRIs to enhance image quality and perform standardized analyses.

### Method
In structural magnetic resonance (MR) imaging, motion artifacts, low resolution, imaging noise, and variability in acquisition protocols, frequently degrade image quality and confound downstream analyses. Here we report a flexible and easy-to-implement Brain MRI Enhancement foundation (BME-X) model for the motion correction, resolution enhancement, denoising and harmonization of MR images. Our framework also exhibits the capability to estimate high-field-like (7T-like) images from 3T images, handle pathological brain MRIs with multiple sclerosis or gliomas, harmonize MRIs acquired by various scanners, and can be easily extended for "end-to-end" neuroimaging analyses, such as tissue segmentation.
<img src="https://github.com/YueSun814/Img-folder/blob/main/BME-X_flowchart.png" width="100%">    

### Motion correction and super resolution:
<div align="center">
<img src="https://github.com/YueSun814/Img-folder/blob/main/BME-X_BCP_real_images.png" width="70%"> 
</div>   
In the first column, from top to bottom are T1w images with severe, moderate and minor artifacts, and two low resolution images (1.0×1.0×3.0 mm<sup>3</sup>). The corresponding enhanced images generated by different methods are shown from the second column to the last column. The resolution of the 2D sagittal and coronal slices (the corrupted images in the last two rows) is indicated in bold as 1.0×1.0×3.0 mm<sup>3</sup> and 1.0×1.0×3.0 mm<sup>3</sup>, respectively. The corresponding enhanced results are in a resolution of 0.8×0.8×0.8 mm<sup>3</sup>.

### Performance on 10,963 lifespan images from fetal to adulthood:
<div align="center">
<img src="https://github.com/YueSun814/Img-folder/blob/main/BME-X_lifespan_real_images.png" width="100%"> 
</div>   
Enhanced results of the BME-X model for 10,963 in vivo low-quality images across the whole human lifespan, collected from 19 datasets. a, Age distribution. b, Mid-late fetal: original T2w images with severe motion and noise, and the corresponding enhanced results. c, From neonatal to early childhood: low-resolution T1w images (1.0×1.0×3.0 mm<sup>3</sup>) and the corresponding enhanced results (0.8×0.8×0.8 mm<sup>3</sup>). d, From neonatal to late adulthood: T1w images and the corresponding enhanced results. e, Comparison of TCT values for the 10,963 original images and the corresponding enhanced images by BME-X. 

### Application on reconstruction 7T-like images from 3T MRIs:
<div align="center">
<img src="https://github.com/YueSun814/Img-folder/blob/main/BME-X_3T_7T.png" width="60%"> 
</div>   

### Application on harmonization across scanners (into a latent common space):
<div align="center">
<img src="https://github.com/YueSun814/Img-folder/blob/main/BME-X_harmonization.png" width="100%"> 
</div>  

### Preservation of small lesions during enhancement:
<div align="center">
<img src="https://github.com/YueSun814/Img-folder/blob/main/BME-X_spots.png" width="50%"> 
</div>  

### Bias quantification during reconstruction:
<div align="center">
<img src="https://github.com/YueSun814/Img-folder/blob/main/BME-X_MR-ART.png" width="100%"> 
</div>  

Enhancement results and the bias quantification for 280 _in vivo_ corrupted T1w images from the [MR-ART dataset](https://openneuro.org/datasets/ds004173/versions/1.0.2), generated by competing methods and the BME-X model. a, Visual comparison of the enhanced results. The first and the seventh columns show _in vivo_ corrupted images with two levels of real artifacts (i.e., HM1 and HM2) acquired from the same participant. The remaining columns show the enhanced results generated by four competing methods and the BME-X model. b, Quantitative comparison on 280 _in vivo_ corrupted images using tissue volumes (i.e., WM, GM, CSF, ventricle, and hippocampus), mean cortical thickness, as well as the corresponding difference compared with STAND. In each box plot, the midline represents the median value, and its lower and upper edges represent the first and third quartiles. The whiskers go down to the smallest value and up to the largest. The dotted line represents the mean value or a zero value. 

Effect size (Cohen’s _d_) of tissue volumes between Alzheimer’s disease and normal cognition participants from the ADNI dataset.

<div align="center">
<img src="https://github.com/YueSun814/Img-folder/blob/main/BME-X_ADNI.jpg" width="80%"> 
</div>  

## Training Steps

   In folder: ***Training_files***
   1. Download an example training dataset (in HDF5 format) from https://www.dropbox.com/scl/fo/8jrphll6vu4sbw56x9ni7/h?rlkey=nfspdxoyr0u61i1xh29dauohu&dl=0. More information about HDF5 format is avaliable at <https://www.mathworks.com/help/matlab/hdf5-files.html>.
   2. Train a caffe model:

    # For a local installation of Caffe
    caffe train -solver solver.prototxt -gpu 0 >train.log 2>&1 &
    
    # For a Docker-based installation of Caffe
    dcaffe train -solver solver.prototxt -gpu 0 >train.log 2>&1 & 

- ***solver.prototxt***: This file defines the training configurations for your model, including:

  - The base learning rate (base_lr: 0.005).
  - The network definition file (net: "train.prototxt").
  - The step size for learning rate adjustments (stepsize: 222). Adjust this parameter based on the number of training samples..
  - The saving path for model snapshots (snapshot_prefix: "./").
  - Other training-related parameters.

- ***-gpu***: Specifies the GPU ID to use for training.

- ***train.log***: A log file that records details of the model's training process.

- ***train.prototxt***: Defines the network architecture.

- ***train_dataset.txt***: Contains the file names of the training dataset.
  
- ***test_dataset.txt***: Contains the file names of the testing dataset.

The network architecture is based on the **Anatomy-Guided Densely-Connected U-Net (ADU-Net)**, as described in the paper:
L. Wang, G. Li, F. Shi, X. Cao, C. Lian, D. Nie, et al., "Volume-based analysis of 6-month-old infant brain MRI for autism biomarker identification and early diagnosis," MICCAI, 2018, pp. 411-419. [Paper](https://liwang.web.unc.edu/wp-content/uploads/sites/11006/2018/10/Volume-Based-Analysis-Of-6-Month-Old.pdf) [Code](https://liwang.web.unc.edu/wp-content/uploads/sites/11006/2020/04/Anatomy_Guided_Densely_Connected_U_Net.txt)

### PyTorch network architecture

   In the folder ***PyTorch_version***, we implemented the same network architecture using PyTorch. The implementation can be found in _Enhancement_model_.py.

## Testing Steps
   
### Folder descriptions
   ***Pretrained_models***: Contains eight pretrained models used for testing images at different ages.
   
   ***Templates***: Includes eight corresponding image templates for histogram matching.
   
   ***Testing_subjects***: A testing sample at 24 months (_test_img.nii_). The corresponding reconstruction result (_test_img-enhanced.nii_).

### How to test
   
   1. Perform histogram matching for testing images using the provided templates (located in the folder ***Templates***):
    
   2. Run the test:
      
    # For a local installation of Caffe:
    python2 BME_X.py --input_path ./Testing_subjects --output_path ./Testing_subjects/output --age_in_month 24
   
    # For a Docker-based installation of Caffe:
    dpython BME_X_docker.py --input_path ./Testing_subjects --output_path ./Testing_subjects/output --age_in_month 24

Please select the corresponding models and reference files based on your requirements:
   -  --input_path Path to the input test images (e.g., ./Testing_subjects/)
   -  --output_path Path to the enhanced images (e.g., ./Testing_subjects/output/)
   -  --age_in_month Age group of the test images (e.g., fetal, 0, 3, 6, 9, 12, 18, 24, adult)
   -  _test_24m.nii_: An example test image at 24 months of age, used as inputs for _Reconstruction_test_docker.py_ and _Reconstruction_test.py_. 
   -  _test_24m-recon.nii_: The corresponding enhanced results for _test_24m.xxx_, generated as output by _Reconstruction_test_docker.py_ and _Reconstruction_test.py_.

## System requirements
    
Caffe==1.0.0-rc3, Python==2.7.17, SimpleITK==1.2.4, numpy==1.13.3, scipy==1.2.3.

We provide two options for installing the Caffe framework on your local PC.  
    
### 1. Docker-based installation of Caffe

Note: The Caffe framework is included in a Docker image for your convenience.

For instructions on Docker and Nvidia-Docker installation, please visit:
https://github.com/iBEAT-V2/iBEAT-V2.0-Docker#run-the-pipeline

#### a. Download the image

Please download the image from the following link: [https://www.dropbox.com/scl/fi/jgszaaldp97fktmi833je/caffe.tar?rlkey=snaxky2fz9ljn8a8mt0jz7d5q&dl=0](https://www.dropbox.com/scl/fi/ulx9a6ytdw2hpeoaahlxt/caffe.tar?rlkey=y6zwg5etyzhbtmw2l2i5rbjk2&dl=0). (The image is _caffe.tar_)

#### b. Load the image into your local PC. 

To load the Docker image, use the following command:

    docker load < caffe.tar 
    
You can verify that the image was loaded successfully by running:

    docker images

#### c. Add an alias to your _~/.bashrc_ file. 

To easily access Caffe from the Docker image, add the following aliases to your ~/.bashrc file: 

    alias dcaffe='nvidia-docker run --rm -u $(id -u):$(id -g) -v $(pwd):$(pwd) -w $(pwd) caffe:v2 /usr/local/caffe/caffe_rc3/build/tools/caffe'
    alias dpython='nvidia-docker run --rm -u $(id -u):$(id -g) -v $(pwd):$(pwd) -w $(pwd) caffe:v2 python'

Then, refresh your shell by running:

    source ~/.bashrc

#### d. Test Caffe  

To test if Caffe is working properly, run:

    dcaffe

The screen will show:  
    
<img src="https://github.com/YueSun814/Img-folder/blob/main/caffe_display.jpg" width="50%">    

    dpython

No output will be displayed. 

### 2. Local installation 

To make sure of consistency with our used version (e.g., including 3d convolution, and WeightedSoftmaxWithLoss, etc.), we strongly recommend installing _Caffe_ using our released ***caffe_rc3***. The installation steps are easy to perform without compilation procedure: 
    
#### a. Download ***caffe_rc3*** and ***caffe_lib***.
    
caffe_rc3: <https://github.com/YueSun814/caffe_rc3>
    
caffe_lib: <https://github.com/YueSun814/caffe_lib>
    
#### b. Add paths of _caffe_lib_, and _caffe_rc3/python_ to your _~/.bashrc_ file. For example, if the folders are saved in the home path, then add the following commands to the _~/.bashrc_ 
   
    export LD_LIBRARY_PATH=~/caffe_lib:$LD_LIBRARY_PATH
   
    export PYTHONPATH=~/caffe_rc3/python:$PATH

Then, "source ~/.bashrc".

#### c. Test Caffe 
    
     cd caffe_rc3/build/tools
    
     ./caffe
    
Then, the screen will show:  
    
<img src="https://github.com/YueSun814/Img-folder/blob/main/caffe_display.jpg" width="50%">
    
Typical install time: few minutes.
  
## License

MIT License
   

    


