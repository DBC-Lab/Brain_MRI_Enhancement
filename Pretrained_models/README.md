## Deploy setting

_deploy.prototxt/fetal_deploy.prototxt_: a deploy file used for testing stage, note that in the testing phase, you have to change "use_global_stats: false" to "use_global_stats: true". 

## Pretrained models:

_reconstruction_??m_T1.caffemodel_: pretrained models used to test T1-weighted images at ?? months. 

_reconstruction_fetal_T2.caffemodel_: pretrained models used to test T2-weighted images at the fetal stage. 

## Adult images: 

Please use _reconstruction_24m_T1.caffemodel_ to test adult T1-weighted images. 
