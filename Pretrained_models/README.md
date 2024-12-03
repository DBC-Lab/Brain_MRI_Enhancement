## Deploy setting

_deploy.prototxt/deploy1.prototxt_: a deploy file used for testing stage, note that in the testing phase, you have to change "use_global_stats: false" to "use_global_stats: true". 

## Pretrained models:

Please download models from [https://www.dropbox.com/scl/fi/ebarde3a4a51bhln3l422/reconstruction_fetal_T2.caffemodel?rlkey=r42d8q4rf57rwhbdg97p3evue&st=v0tjodhh&dl=0 ](https://www.dropbox.com/scl/fo/jkaoez96guhnnd39qmt96/ACc8JoYPflw_pDpjKvv1-dw?rlkey=904z1tlyw2dhb3b76m25s4utt&st=wgfp703y&dl=0)

_reconstruction_??m_T1.caffemodel_: pretrained models used to test T1-weighted images at ?? months. 

_reconstruction_fetal_T2.caffemodel_: pretrained models used to test T2-weighted images at the fetal stage. 

## Adult images: 

Please use _reconstruction_24m_T1.caffemodel_ to test adult T1-weighted images. 
