# Lifespan Brain MRI Enhancement for Motion Removal, Super Resolution, Denoising and Multiple Extensions

### Structural magnetic resonance (MR) imaging is a vital tool for neuroimaging analyses, but the quality of MR images is often degraded by various factors, such as motion artifacts, large slice thickness, and imaging noise. These factors can cause confounding effects and pose significant challenges, especially in young children who tend to move during acquisition.
<img src="https://github.com/YueSun814/Img-folder/blob/main/Flowchart_Reconstruction.jpg" width="100%">

## Method
This manuscript describes a flexible and easy-to-implement method for significantly improving the quality of brain MR images through motion removal, super resolution, and denoising. 

## Training steps:

   In folder: ***Training_files***
   1. Download training samples (hdf5 data) from https://www.dropbox.com/scl/fo/8jrphll6vu4sbw56x9ni7/h?rlkey=nfspdxoyr0u61i1xh29dauohu&dl=0. More information about hdf5 data is avaliable at <https://www.mathworks.com/help/matlab/hdf5-files.html>.
   2. Train a caffe model:
      
    caffe train -solver solver.prototxt -gpu 0 >train.log 2>&1 &

## Testing steps:

   1. In folder ***Pretrained_models***: eight pretrained models used for testing images at different ages.
   2. In folder ***Templates***: the corresponding eight images templates for histogram matching.
   3. In folder ***Testing_subjects***: a testing sample at 24 months (test_img.???), and the corresponding reconstruction result (test_img-recon.nii.gz) after performing step 4.
   4. Test an image: 
      
    python Reconstruction_T1_test.py    
    
   
## System requirements:

Ubuntu 20.04.1
    
Caffe version 1.0.0-rc3
    
To make sure of consistency with our used version (e.g., including 3d convolution, and WeightedSoftmaxWithLoss, etc.), we strongly recommend installing _Caffe_ using our released ***caffe_rc3***. The installation steps are easy to perform without compilation procedure: 
    
### a. Download ***caffe_rc3*** and ***caffe_lib***.
    
caffe_rc3: <https://github.com/YueSun814/caffe_rc3>
    
caffe_lib: <https://github.com/YueSun814/caffe_lib>
    
### b. Add paths of _caffe_lib_, and _caffe_rc3/python_ to your _~/.bashrc_ file. For example, if the folders are saved in the home path, then add the following commands to the _~/.bashrc_ 
   
    export LD_LIBRARY_PATH=~/caffe_lib:$LD_LIBRARY_PATH
   
    export PYTHONPATH=~/caffe_rc3/python:$PATH
    
### c. Test Caffe 
    
     cd caffe_rc3/build/tools
    
     ./caffe
    
Then, the screen will show:  
    
<img src="https://github.com/YueSun814/Img-folder/blob/main/caffe_display.jpg" width="50%">
    
Typical install time: few minutes.
   

    


