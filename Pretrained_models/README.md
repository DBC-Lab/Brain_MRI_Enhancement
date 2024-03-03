## Deploy setting

***deploy.prototxt***: a deploy file used for testing stage, note that in the testing phase, you have to change "use_global_stats: false" to "use_global_stats: true". 

## Pretrained models:

reconstruction_??m_T1.caffemodel: pretrained models used to test T1-weighted images at ?? months. 

reconstruction_fetal_T2.caffemodel: pretrained models used to test T2-weighted images at the fetal stage. 

## Adult images: 

Please use ***reconstruction_24m_T1.caffemodel*** to test adult T1-weighted images. 
