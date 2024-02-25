# Brain MRI Enhancement for Motion Removal, Super Resolution, Denoising and Multiple Extensions

### Structural magnetic resonance (MR) imaging is a vital tool for neuroimaging analyses, but the quality of MR images is often degraded by various factors, such as motion artifacts, large slice thickness, and imaging noise. These factors can cause confounding effects and pose significant challenges, especially in young children who tend to move during acquisition.
<img src="https://github.com/YueSun814/Img-folder/blob/main/Flowchart_Reconstruction.jpg" width="100%">

## Method
This manuscript describes a flexible and easy-to-implement method for significantly improving the quality of brain MR images through motion removal, super resolution, and denoising. 

### Training steps:

   In folder: ***Training_files***
   

   
### System requirements:

    Ubuntu 18.04.5
    
    Caffe version 1.0.0-rc3
    
    To make sure of consistency with our used version (e.g., including 3d convolution, and WeightedSoftmaxWithLoss, etc.), we strongly recommend installing _Caffe_ using our released ***caffe_rc3***. The installation steps are easy to perform without compilation procedure: 
    
    a. Download ***caffe_rc3*** and ***caffe_lib***.
    
    caffe_rc3: <https://github.com/YueSun814/caffe_rc3>
    
    caffe_lib: <https://github.com/YueSun814/caffe_lib>
    
    b. Add paths of _caffe_lib_, and _caffe_rc3/python_ to your _~/.bashrc_ file. For example, if the folders are saved in the home path, then add the following commands to the _~/.bashrc_ 
   
   `export LD_LIBRARY_PATH=~/caffe_lib:$LD_LIBRARY_PATH`
   
   `export PYTHONPATH=~/caffe_rc3/python:$PATH`
    
    c. Test Caffe 
    
    `cd caffe_rc3/build/tools`
    
    `./caffe`
    
    Then, the screen will show:  
    
    <img src="https://github.com/YueSun814/Img-folder/blob/main/caffe_display.jpg" width="50%">
    
    Typical install time: few minutes.
   

    


