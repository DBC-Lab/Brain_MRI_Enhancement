## Training samples:

Please download training datasets (*.hdf5) from https://www.dropbox.com/scl/fo/8jrphll6vu4sbw56x9ni7/h?rlkey=nfspdxoyr0u61i1xh29dauohu&dl=0.

## To train a caffe model, please use this command: 

    caffe train -solver solver.prototxt -gpu 0 >train.log 2>&1 &   # for local installtion of Caffe
    
    dcaffe train -solver solver.prototxt -gpu 0 >train.log 2>&1 &   # for Docker installtion of Caffe

***solver.prototxt***: set your learning rate (base_lr: 0.005), network (net: "train.prototxt"), step size (stepsize=222),  saving path (snapshot_prefix: "./"), etc.

***-gpu***: set GPU number.

***train.log***: a log file to record the model training stage.

***train.prototxt***: network architecture.

This is the architecture based on Anatomy-Guided Densely-Connected U-Net (ADU-Net) in the paper of "L. Wang, G. Li, F. Shi, X. Cao, C. Lian, D. Nie, et al., "Volume-based analysis of 6-month-old infant brain MRI for autism biomarker identification and early diagnosis," in MICCAI, 2018, pp. 411-419."

