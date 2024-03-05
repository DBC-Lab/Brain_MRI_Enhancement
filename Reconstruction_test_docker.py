import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np  
import scipy.io as scio
from scipy import ndimage as nd

# Make sure that caffe is on the python path:
caffe_root = '/usr/local/InfantPipeline/lib/caffe/'  # Make sure that caffe is on the python path  
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
dFA=[d1,d2,d3]  #patch size: 40*40*40
dSeg=[40,40,40]  #equal or smaller than patch size, to avoid margin issue when testing 

#step size, usually set as [8, 16]
step1=12
step2=12
step3=12

step=[step1,step2,step3]
NumOfClass=4 #the number of classes in this segmentation project, e.g., WM, GM, CSF and background in this case
    
def cropCubic(matFA,fileID,d,step,rate):
    eps=1e-5
    #transpose
    matFA=np.transpose(matFA,(0,2,1))

    [row,col,leng]=matFA.shape
    margin1=(dFA[0]-dSeg[0])/2
    margin2=(dFA[1]-dSeg[1])/2
    margin3=(dFA[2]-dSeg[2])/2
    cubicCnt=0
    marginD=[margin1,margin2,margin3]
    
    print 'matFA shape is ',matFA.shape
    matFAOut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    print 'matFAOut shape is ',matFAOut.shape
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA


    if margin1!=0:
        matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[matFA.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,matFA.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,matFA.shape[2]-1:leng-marginD[2]-1:-1]
  
    matFAOutScale = nd.interpolation.zoom(matFAOut, zoom=rate)

    matOut=np.zeros((matFA.shape[0],matFA.shape[1],matFA.shape[2],NumOfClass))
    heatmap=np.zeros((matFA.shape[0],matFA.shape[1],matFA.shape[2]))
  
    Visit=np.zeros((matFA.shape[0],matFA.shape[1],matFA.shape[2]))+eps
    [row,col,leng]=matFA.shape
        
    for i in range(d[0]/2+marginD[0]+1,row-d[0]/2-marginD[0]-2,step[0]):
        for j in range(d[1]/2+marginD[1]+1,col-d[1]/2-marginD[1]-2,step[1]):
            for k in range(d[2]/2+marginD[2]+1,leng-d[2]/2-marginD[2]-2,step[2]):
                volFA=matFA[i-d[0]/2-marginD[0]:i+d[0]/2+marginD[0],j-d[1]/2-marginD[1]:j+d[1]/2+marginD[1],k-d[2]/2-marginD[2]:k+d[2]/2+marginD[2] ]
                
                if np.sum(volFA)>10 :

                    volFA=np.float64(volFA)

                    #print 'volFA shape is ',volFA.shape
                    mynet.blobs['dataT1'].data[0,0,...]=volFA

                    mynet.forward()
                    temppremat = mynet.blobs['conv6_3-BatchNorm1'].data #Note you have add softmax layer in deploy prototxt
                    Visit[i-d[0]/2:i+d[0]/2,j-d[1]/2:j+d[1]/2,k-d[2]/2:k+d[2]/2]=Visit[i-d[0]/2:i+d[0]/2,j-d[1]/2:j+d[1]/2,k-d[2]/2:k+d[2]/2]+1
                    heatmap[i-d[0]/2:i+d[0]/2,j-d[1]/2:j+d[1]/2,k-d[2]/2:k+d[2]/2]=heatmap[i-d[0]/2:i+d[0]/2,j-d[1]/2:j+d[1]/2,k-d[2]/2:k+d[2]/2]+temppremat[0,0,marginD[0]:marginD[0]+d[0],marginD[1]:marginD[1]+d[1],marginD[2]:marginD[2]+d[2]]

    heatmap = heatmap/Visit
    heatmap=np.transpose(heatmap,(0,2,1))
    
    return heatmap
 
#this function is used to compute the dice ratio
def dice(im1, im2,tid):
    im1=im1==tid #make it boolean
    im2=im2==tid #make it boolean
    im1=np.asarray(im1).astype(np.bool)
    im2=np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc=2. * intersection.sum() / (im1.sum() + im2.sum())
    return dsc

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
        heatmap = cropCubic(mrimgT1,fileID,dSeg,step,rate)
        
        volOut=sitk.GetImageFromArray(heatmap)
	volOut.SetSpacing([0.8,0.8,0.8])
        sitk.WriteImage(volOut,'./{}/{}-recon.nii.gz'.format(datapath, myid))   


if __name__ == '__main__':     
    main()