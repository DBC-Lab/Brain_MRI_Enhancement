import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np  
import scipy.io as scio
from scipy import ndimage as nd

# Make sure that caffe is on the python path:
caffe_root = '~/caffe_rc3/'  # Make sure that caffe is on the python path  
import sys
sys.path.insert(0, caffe_root + 'python')
#print(caffe_root + 'python')
import caffe

caffe.set_device(0) #set your GPU number
caffe.set_mode_gpu()
### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
protopath='Pretrained_models/'    #the path to accesss pretrained models and deploy.prototxt

#change pretrained models to match the age of test image, "reconstruction_24m_T1.caffemodel" is used to test images at 24 months and older
mynet = caffe.Net(protopath+'deploy.prototxt',protopath+'reconstruction_24m_T1.caffemodel',caffe.TEST)     
print("blobs {}\nparams {}".format(mynet.blobs.keys(), mynet.params.keys()))

d1=40
d2=40
d3=40
dPatchT1=[d1,d2,d3]  #patch size: 40*40*40
dPatchT1_ROI=[40,40,40]  #equal or smaller than patch size, to avoid margin issue when testing 

#step size, usually set as [8, 16]: small step size results in good results but at the cost of long running time.
step1=12
step2=12
step3=12

step=[step1,step2,step3]
NumOfClass=4 #the number of classes in this segmentation project, e.g., WM, GM, CSF and background in this case
    
def cropCubic(matT1,fileID,d,step,rate):
    eps=1e-5
    #transpose
    matT1=np.transpose(matT1,(0,2,1))

    [row,col,leng]=matT1.shape
    margin1=(dPatchT1[0]-dPatchT1_ROI[0])/2
    margin2=(dPatchT1[1]-dPatchT1_ROI[1])/2
    margin3=(dPatchT1[2]-dPatchT1_ROI[2])/2
    cubicCnt=0
    marginD=[margin1,margin2,margin3]
    
    print 'matT1 shape is ',matT1.shape
    matT1Out=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    print 'matT1Out shape is ',matT1Out.shape
    matT1Out[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matT1


    if margin1!=0:
        matT1Out[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matT1[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matT1Out[row+marginD[0]:matT1Out.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matT1[matT1.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matT1Out[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matT1[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matT1Out[marginD[0]:row+marginD[0],col+marginD[1]:matT1Out.shape[1],marginD[2]:leng+marginD[2]]=matT1[:,matT1.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matT1Out[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matT1[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matT1Out[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matT1Out.shape[2]]=matT1[:,:,matT1.shape[2]-1:leng-marginD[2]-1:-1]
  
    matT1OutScale = nd.interpolation.zoom(matT1Out, zoom=rate)

    matOut=np.zeros((matT1.shape[0],matT1.shape[1],matT1.shape[2],NumOfClass))
    Recon=np.zeros((matT1.shape[0],matT1.shape[1],matT1.shape[2]))
  
    Visit=np.zeros((matT1.shape[0],matT1.shape[1],matT1.shape[2]))+eps
    [row,col,leng]=matT1.shape
        
    for i in range(d[0]/2+marginD[0]+1,row-d[0]/2-marginD[0]-2,step[0]):
        for j in range(d[1]/2+marginD[1]+1,col-d[1]/2-marginD[1]-2,step[1]):
            for k in range(d[2]/2+marginD[2]+1,leng-d[2]/2-marginD[2]-2,step[2]):
                volPatchT1=matT1[i-d[0]/2-marginD[0]:i+d[0]/2+marginD[0],j-d[1]/2-marginD[1]:j+d[1]/2+marginD[1],k-d[2]/2-marginD[2]:k+d[2]/2+marginD[2] ]
                
                if np.sum(volPatchT1)>10 :

                    volPatchT1=np.float64(volPatchT1)

                    #print 'volPatchT1 shape is ',volPatchT1.shape
                    mynet.blobs['dataT1'].data[0,0,...]=volPatchT1

                    mynet.forward()
                    temppremat = mynet.blobs['conv6_3-BatchNorm1'].data #Note you have add softmax layer in deploy prototxt
                    Visit[i-d[0]/2:i+d[0]/2,j-d[1]/2:j+d[1]/2,k-d[2]/2:k+d[2]/2]=Visit[i-d[0]/2:i+d[0]/2,j-d[1]/2:j+d[1]/2,k-d[2]/2:k+d[2]/2]+1
                    Recon[i-d[0]/2:i+d[0]/2,j-d[1]/2:j+d[1]/2,k-d[2]/2:k+d[2]/2]=Recon[i-d[0]/2:i+d[0]/2,j-d[1]/2:j+d[1]/2,k-d[2]/2:k+d[2]/2]+temppremat[0,0,marginD[0]:marginD[0]+d[0],marginD[1]:marginD[1]+d[1],marginD[2]:marginD[2]+d[2]]

    Recon = Recon/Visit
    Recon=np.transpose(Recon,(0,2,1))
    
    return Recon

def main():
    datapath='Testing_subjects/' #the path to your test images

    files=[i for i in os.listdir(datapath) if '.hdr' in i ]
    for dataT1filename in files:
        myid=dataT1filename[0:len(dataT1filename)-4]
        fileID='%s'%myid
        dataT1fn=os.path.join(datapath,dataT1filename)
        print dataT1fn
        imgOrg=sitk.ReadImage(dataT1fn)
        mrimgT1=sitk.GetArrayFromImage(imgOrg)

        rate=1
        Recon = cropCubic(mrimgT1,fileID,dPatchT1_ROI,step,rate)
        
        volOut=sitk.GetImageFromArray(Recon)
	volOut.SetSpacing([0.8,0.8,0.8])
        sitk.WriteImage(volOut,'./{}/{}-recon.nii.gz'.format(datapath, myid))   


if __name__ == '__main__':     
    main()
