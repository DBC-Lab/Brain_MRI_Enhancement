# Lifespan Brain MRI Enhancement for Motion Removal, Super Resolution, Denoising and More

### Structural magnetic resonance (MR) imaging is a vital tool for neuroimaging analyses, but the quality of MR images is often degraded by various factors, such as motion artifacts, large slice thickness, and imaging noise. These factors can cause confounding effects and pose significant challenges, especially in young children who tend to move during acquisition.
<img src="https://github.com/YueSun814/Img-folder/blob/main/Flowchart_Reconstruction.png" width="100%">

## Method
This manuscript describes a novel tissue-aware reconstruction framework that can improve image quality through motion correction, super-resolution, denoising, and contrast enhancement. our framework exhibits the capability to estimate high-field-like (7T-like) images from 3T images, handle pathological brain MRIs with multiple sclerosis or gliomas, harmonize MRIs acquired by various scanners, and can be easily extended for "end-to-end" neuroimaging analyses, such as tissue segmentation.

## Training steps:

   In folder: ***Training_files***
   1. Download training samples (hdf5 data) from https://www.dropbox.com/scl/fo/8jrphll6vu4sbw56x9ni7/h?rlkey=nfspdxoyr0u61i1xh29dauohu&dl=0. More information about hdf5 data is avaliable at <https://www.mathworks.com/help/matlab/hdf5-files.html>.
   2. Train a caffe model:
      
    caffe train -solver solver.prototxt -gpu 0 >train.log 2>&1 &  # for local installtion of Caffe
    
    dcaffe train -solver solver.prototxt -gpu 0 >train.log 2>&1 &  # for Docker installtion of Caffe

   ***solver.prototxt***: set your learning rate (base_lr: 0.005), network (net: "train.prototxt"), step size (stepsize=222),  saving path (snapshot_prefix: "./"), etc.

   ***-gpu***: set GPU number.

   ***train.log***: a log file to record the model training stage.

   ***train.prototxt***: network architecture.

   This is the architecture based on Anatomy-Guided Densely-Connected U-Net (ADU-Net) in the paper of "L. Wang, G. Li, F. Shi, X. Cao, C. Lian, D. Nie, et al., "Volume-based analysis of 6-month-old infant brain MRI for autism biomarker identification and early diagnosis," in MICCAI, 2018, pp. 411-419."

## Testing steps:

   In folder ***Pretrained_models***: eight pretrained models used for testing images at different ages.
   
   In folder ***Templates***: the corresponding eight images templates for histogram matching.
   
   In folder ***Testing_subjects***: a testing sample at 24 months (test_img.???), and the corresponding reconstruction result (test_img-recon.nii.gz).
   
   Test an image: 
   
   1. Performing histogram matching for testing images with provided templates (in folder ***Templates***). 
    
   2. Test:
      
    python2 Reconstruction_test.py   # for local installtion of Caffe
    
    dpython Reconstruction_test_docker.py  # for Docker installtion of Caffe
    
## System requirements:

Ubuntu 20.04.1
    
Caffe version 1.0.0-rc3

We provided two options for installing the Caffe framework on your local PC.  
    
### 1. Docker version

!!! We have included the Caffe framework in a Docker image. 

#### a. Download an image. 

Please download the image from the following link: https://www.dropbox.com/scl/fi/jgszaaldp97fktmi833je/caffe.tar?rlkey=snaxky2fz9ljn8a8mt0jz7d5q&dl=0. (The image is _caffe.tar_)

#### b. Load the image into your local PC. 

    docker load < caffe.tar 

Then you can check the image using this command "docker images".  

#### c. Add an alias to your _~/.bashrc_ file. to easily access Caffe from the Docker image. 

    alias dcaffe='nvidia-docker run --rm -u $(id -u):$(id -g) -v $(pwd):$(pwd) -w $(pwd) caffe:v2 /usr/local/caffe/caffe_rc3/build/tools/caffe'
    alias dpython='nvidia-docker run --rm -u $(id -u):$(id -g) -v $(pwd):$(pwd) -w $(pwd) caffe:v2 python'

Then, "source ~/.bashrc". 

#### d. Test Caffe  

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
    
#### c. Test Caffe 
    
     cd caffe_rc3/build/tools
    
     ./caffe
    
Then, the screen will show:  
    
<img src="https://github.com/YueSun814/Img-folder/blob/main/caffe_display.jpg" width="50%">
    
Typical install time: few minutes.
   

    


