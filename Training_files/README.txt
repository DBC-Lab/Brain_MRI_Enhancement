Please download training datasets (*.hdf5) from https://www.dropbox.com/scl/fo/8jrphll6vu4sbw56x9ni7/h?rlkey=nfspdxoyr0u61i1xh29dauohu&dl=0.

To train a caffe model, please use this command: caffe train -solver solver.prototxt -gpu 0 >train.log 2>&1 & 
*** solver.prototxt: set up your own learning rate, network, saving path, etc.
*** -gpu: set up your gpu number
*** train.log: a log file to record the model training stage
