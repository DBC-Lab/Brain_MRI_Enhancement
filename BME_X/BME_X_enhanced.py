import SimpleITK as sitk
import os
import numpy as np
import torch
from BME_X.options.train_options import TrainOptions
from BME_X.models.models import create_model
import argparse
from tqdm import tqdm
from BME_X.models.DUNet3D_seg_recon_softmax import DenseUNet3d

def find_boundary_img(Img, margin1, margin2, margin3):
    [Height, Wide, Z] = Img.shape
    for i in range(0, Height - 1, 1):
        temp = Img[i, :, :]
        if sum(sum(temp[:])) > 0:
            a = i
            break

    for i in range(Height - 1, 0, -1):
        temp = Img[i, :, :]
        if sum(sum(Img[i, :, :])) > 0:
            b = i
            break
    for i in range(0, Wide - 1, 1):
        temp = Img[:, i, :]
        if (sum(sum(temp[:])) > 0):
            c = i
            break

    for i in range(Wide - 1, 0, -1):
        temp = Img[:, i, :]
        if (sum(sum(temp[:])) > 0):
            dd = i
            break

    for i in range(0, Z - 1, 1):
        temp = Img[:, :, i]
        if (sum(sum(temp[:])) > 0):
            e = i
            break

    for i in range(Z - 1, 0, -1):
        temp = Img[:, :, i]
        if (sum(sum(temp[:])) > 0):
            f = i
            break

    if (a - margin1 / 2 <= 0):
        a = margin1 + 1

    if (c - margin2 / 2 <= 0):
        c = margin2 + 1

    if (e - margin3 / 2 <= 0):
        e = margin3 + 1

    if (b + margin1 >= Height):
        b = Height - margin1

    if (dd + margin2 >= Wide):
        dd = Wide - margin2

    if (f + margin3 >= Z):
        f = Z - margin3

    return a, b, c, dd, e, f


def hist_match(img, temp):
    ''' histogram matching from img to temp '''
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    res = matcher.Execute(img, temp)
    return res


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")

def resample(resampler, processed_image,  new_spacing, new_size, original_spacing, original_size, original_origin, original_direction):
    processed_image = sitk.GetImageFromArray(processed_image)
    processed_image.SetSpacing(new_spacing)
    processed_image.SetOrigin(original_origin)
    processed_image.SetDirection(original_direction)
    
    resampled_back_image = resampler.Execute(processed_image)
    
    resampled_back_image.SetSpacing(original_spacing)
    resampled_back_image.SetOrigin(original_origin)
    resampled_back_image.SetDirection(original_direction)
    resampled_back_image = sitk.GetArrayFromImage(resampled_back_image)
    
    return resampled_back_image

def cropCubic_first(matT1_ori, new_spacing, new_size, original_spacing,original_size,original_origin, original_direction,matT1_v0, fileID, device,model):
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(original_spacing)
    resampler.SetSize(original_size)
    resampler.SetOutputDirection(original_direction)
    resampler.SetOutputOrigin(original_origin)
    resampler.SetInterpolator(sitk.sitkLinear)

    matT1=np.transpose(matT1_v0,(0,2,1))
    
    margin=5
    d1 = 40
    d2 = 40
    d3 = 40
    dFA = [d1, d2, d3]  # size of patches of input data
    d = dFA
    step1 = 18
    step2 = 18
    step3 = 18
    step = [step1, step2, step3]

    # the number of classes in this segmentation project
    NumOfClass = 4  # bg, wm, gm, csf
    
    eps = 1e-5
    [row, col, leng] = matT1.shape
    matSegScale = matT1

    Visit = np.zeros((matSegScale.shape[0]+2*dFA[0], matSegScale.shape[1]+2*dFA[1], matSegScale.shape[2]+2*dFA[2]))+eps
    
    MROut = np.zeros((matT1.shape[0], matT1.shape[2], matT1.shape[1]))
    matOut_extend = np.zeros((matSegScale.shape[0]+2*dFA[0], matSegScale.shape[1]+2*dFA[1], matSegScale.shape[2]+2*dFA[2], NumOfClass))
    MROut_extend = np.zeros((matSegScale.shape[0]+2*dFA[0], matSegScale.shape[1]+2*dFA[1], matSegScale.shape[2]+2*dFA[2]))
    PrOut = np.zeros((matSegScale.shape[0], matSegScale.shape[1], matSegScale.shape[2], NumOfClass))
    #PrOut1 = np.zeros((matSegScale.shape[0], matSegScale.shape[2], matSegScale.shape[1], NumOfClass))
    PrOut_ori = np.zeros((matT1_ori.shape[0], matT1_ori.shape[1], matT1_ori.shape[2], NumOfClass))
    PrOut_extend = np.zeros((matSegScale.shape[0]+2*dFA[0], matSegScale.shape[1]+2*dFA[1], matSegScale.shape[2]+2*dFA[2], NumOfClass))
    matSegScale_extend = np.zeros((matSegScale.shape[0]+2*dFA[0], matSegScale.shape[1]+2*dFA[1], matSegScale.shape[2]+2*dFA[2]))
    matSegScale_extend[dFA[0]:matSegScale.shape[0]+dFA[0], dFA[1]:matSegScale.shape[1]+dFA[1], dFA[2]:matSegScale.shape[2]+dFA[2]] = matSegScale

    [aa, bb, cc, dd, ee, ff] = find_boundary_img(matSegScale_extend, dFA[0]+margin, dFA[1]+margin, dFA[2]+margin)
    volume = np.zeros((1, 1, dFA[0], dFA[1], dFA[2]))
    
    total_iterations = ((bb + margin - (aa - margin)) // step[0]) * ((dd + margin - (cc - margin)) // step[1])* ((ee + margin - (ff - margin)) // step[2])
    
    print('Processing ...')
    with tqdm(total=total_iterations) as pbar:
        for i in range(aa-margin, bb+margin, step[0]):
            for j in range(cc-margin, dd+margin, step[1]):
                for k in range(ee-margin, ff+margin, step[2]):
                    pbar.update(1)
                    volume[0, 0, :, :, :] = matSegScale_extend[i:i + d[0], j:j + d[1], k:k + d[2]]
                    volMR1 = torch.tensor(volume)  # patch
                    volMR = volMR1.to(device, dtype=torch.float)
                    seg, recon = model(volMR)
                    tempprob = (torch.squeeze(seg)).cpu().detach().numpy()
                    matRecon = (torch.squeeze(recon)).cpu().detach().numpy()
                    MROut_extend[i+margin:i + d[0]-margin, j+margin:j + d[1]-margin, k+margin:k + d[2]-margin] = MROut_extend[i+margin:i + d[0]-margin, j+margin:j + d[1]-margin, k+margin:k + d[2]-margin] + matRecon[margin:d[0]-margin, margin:d[1]-margin, margin:d[2]-margin]
                    Visit[i+margin:i + d[0]-margin, j+margin:j + d[1]-margin, k+margin:k + d[2]-margin] = Visit[i+margin:i + d[0]-margin, j+margin:j + d[1]-margin, k+margin:k + d[2]-margin] + 1

                    temppremat = tempprob.argmax(axis=0)  # Note you have add softmax layer in deploy prototxt

                    for labelInd in range(NumOfClass):
                        currLabelMat = np.where(temppremat == labelInd, 1,
                                                0)  # if satisfy the condition (labelInd), then output 1; else 0
                        matOut_extend[i+margin:i + d[0]-margin, j+margin:j + d[1]-margin, k+margin:k + d[2]-margin, labelInd] = matOut_extend[i+margin:i + d[0]-margin, j+margin:j + d[1]-margin, k+margin:k + d[2]-margin,
                                                                           labelInd] + currLabelMat[margin:d[0]-margin, margin:d[1]-margin, margin:d[2]-margin]
                        PrOut_extend[i+margin:i + d[0]-margin, j+margin:j + d[1]-margin, k+margin:k + d[2]-margin, labelInd] = PrOut_extend[i+margin:i + d[0]-margin, j+margin:j + d[1]-margin, k+margin:k + d[2]-margin,
                                                                          labelInd] + tempprob[labelInd, margin:d[0]-margin, margin:d[1]-margin, margin:d[2]-margin]

    PrOut = PrOut_extend[dFA[0]:matSegScale.shape[0]+dFA[0], dFA[1]:matSegScale.shape[1]+dFA[1], dFA[2]:matSegScale.shape[2]+dFA[2],:]
    Visit_ori = Visit[dFA[0]:matSegScale.shape[0]+dFA[0], dFA[1]:matSegScale.shape[1]+dFA[1], dFA[2]:matSegScale.shape[2]+dFA[2]]
    MROut = MROut_extend[dFA[0]:matSegScale.shape[0]+dFA[0], dFA[1]:matSegScale.shape[1]+dFA[1], dFA[2]:matSegScale.shape[2]+dFA[2]]
    
    MROut = MROut/Visit_ori
    label1 = PrOut[:,:,:,0]/Visit_ori
    label2 = PrOut[:,:,:,1]/Visit_ori
    label3 = PrOut[:,:,:,2]/Visit_ori
    label4 = PrOut[:,:,:,3]/Visit_ori

    label1 = np.transpose(label1, (0, 2, 1))
    label2 = np.transpose(label2, (0, 2, 1))
    label3 = np.transpose(label3, (0, 2, 1))
    label4 = np.transpose(label4, (0, 2, 1))
    MROut = np.transpose(MROut, (0, 2, 1))
    
    label1_ori = resample(resampler, label1, new_spacing, new_size, original_spacing, original_size, original_origin, original_direction)
    label2_ori = resample(resampler, label2, new_spacing, new_size, original_spacing, original_size, original_origin, original_direction)
    label3_ori = resample(resampler, label3, new_spacing, new_size, original_spacing, original_size, original_origin, original_direction)
    label4_ori = resample(resampler, label4, new_spacing, new_size, original_spacing, original_size, original_origin, original_direction)
    MROut_ori = resample(resampler, MROut, new_spacing, new_size, original_spacing, original_size, original_origin, original_direction)

    PrOut_ori[:, :, :, 0] = label1_ori
    PrOut_ori[:, :, :, 1] = label2_ori
    PrOut_ori[:, :, :, 2] = label3_ori
    PrOut_ori[:, :, :, 3] = label4_ori
    
    matOut_ori = PrOut_ori.argmax(axis=3)  # always 3
    matOut_ori = np.rint(
        matOut_ori) 
    
    
    return MROut, MROut_ori, matOut_ori

def calculate_tct(mean_A, mean_B, std_A, std_B):
    """
    Calculate Tissue Contrast T-score (TCT) between two tissues.
    
    Parameters:
    mean_A (float): Mean intensity of tissue A
    mean_B (float): Mean intensity of tissue B
    std_A (float): Standard deviation of intensity of tissue A
    std_B (float): Standard deviation of intensity of tissue B
    
    Returns:
    float: TCT score
    """
    tct = abs(mean_A - mean_B) / np.sqrt(std_A**2 + std_B**2)
    return tct

def load_and_cast_image(filepath, dtype=sitk.sitkFloat32):
    """Load an image and cast it to the specified data type."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    img = sitk.ReadImage(filepath)
    return sitk.Cast(img, dtype)

def BMEX(input_path, output_path, age_number, suffix, device):
    datapath = input_path
    datapath_output = output_path

    if age_number == 'adult':
        age_number = '24'
    elif age_number == 'fetal':
        age_number = 'Fetal'
    else:
        age_number = int(age_number)
        if age_number >= 21:
            age_number = '24'
        elif 15 <= age_number < 21:
            age_number = '18'
        elif 10 <= age_number < 15:
            age_number = '12'
        elif 7 <= age_number < 10:
            age_number = '9'
        elif 5 <= age_number < 7:
            age_number = '6'
        elif 2 <= age_number < 5:
            age_number = '3'
        elif -0.00001 <= age_number < 2:
            age_number = '0'
        elif age_number < 0:
            age_number = 'Fetal'
        else:
            print('No age information, please add it.')

    gpu_device = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # opt = TrainOptions().parse()
    # model = create_model(opt)
    model = DenseUNet3d()
    # model.load_state_dict(torch.load('./checkpoints/Tissue-Month24-T1-model.pt',map_location=device))

    # change pretrained models to match the age of test image, "BMEX-Month24-T1-model.pt" is used to test images at 24 months and older
    if age_number == 'fetal':
        model.load_state_dict(
            torch.load('/BME_X/checkpoints/BMEX-Fetal-T2-model.pt', map_location=device, weights_only=True))
        print("Model: BMEX-fetal-T2-model.pt")
    else:
        if suffix == 'T1w':
            model_file = '/BME_X/checkpoints/BMEX-Month{}-T1-model.pt'.format(age_number)
            model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
        elif suffix == 'T2w':
            model_file = '/BME_X/checkpoints/BMEX-Month{}-T2-model.pt'.format(age_number)
            model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
        print("Model: ", model_file)

    model.to(device)
    model.eval()  # Set the model to evaluation mode



    if age_number=='fetal':
        reference_data = sitk.ReadImage('/BME_X/Templates/Template-Fetal-T2w.nii.gz')
        print("Reference image: Template-Fetal-T2w.nii.gz")
        files=[i for i in os.listdir(datapath) if 'T2w.nii.gz' in i ]
        
    else: 
        reference_data = sitk.ReadImage('/BME_X/Templates/Template-Month{}-{}.nii.gz'.format(age_number, suffix)) #reference file for histogram matching, e.g., Template-Month24-T1.nii.gz is the reference file for testing images at 24 months and older.
        print("Reference image: Template-Month{}-{}.nii.gz".format(age_number, suffix))
        if suffix=='T1w':
            files=[i for i in os.listdir(datapath) if 'T1w-reorient-n4.nii.gz' in i ]
        elif suffix=='T2w':
            files=[i for i in os.listdir(datapath) if 'T2w-reorient-n4.nii.gz' in i ]
    
    reference_data = sitk.Cast(reference_data, sitk.sitkFloat32)
    
    #files=[i for i in os.listdir(datapath) if '.nii' in i ]
    for dataT1filename in files:
        subject_id = dataT1filename[:-10]
        dataT1fn=os.path.join(datapath,dataT1filename)
        
        # Load original MRI image
        dataT1fn = os.path.join(datapath, dataT1filename)
        imgOrg = load_and_cast_image(dataT1fn)
        mrimg_ori = sitk.GetArrayFromImage(imgOrg)
    
        # Load brain mask
        datamaskfn = os.path.join(datapath, f'{subject_id}-brainmask.nii.gz')
        imgOrg_mask = load_and_cast_image(datamaskfn)
        maskimg = sitk.GetArrayFromImage(imgOrg_mask)
    
        # Apply mask
        mrimg = mrimg_ori * maskimg
        filenameT1 = sitk.GetImageFromArray(mrimg)
        filenameT1.SetOrigin(imgOrg.GetOrigin())
        filenameT1.SetDirection(imgOrg.GetDirection())
        filenameT1.SetSpacing(imgOrg.GetSpacing())
        pixelID = filenameT1.GetPixelID()
        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(pixelID)
        filenameT1_forhist = filenameT1

        # Apply N4 bias correction if specified
        filenameT1 = sitk.Cast(filenameT1, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
        filenameT1 = corrector.Execute(filenameT1)
        mrimg = sitk.GetArrayFromImage(filenameT1)

        
        # Save masked image
        #sitk.WriteImage(filenameT1, os.path.join(datapath_output, f'{subject_id}-brain.nii.gz'))
    
        # Resample image
        print('Resampling images ...')
        original_spacing = imgOrg.GetSpacing()
        new_spacing = (0.75, 0.75, 0.75) if age_number == 'fetal' else (0.8, 0.8, 0.8)
    
        original_size = imgOrg.GetSize()
        new_size = [
            int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
            for i in range(3)
        ]
    
        print('Original size:', original_size)
        print('New size:', new_size)
        print('Original spacing:', original_spacing)
        print('New spacing:', new_spacing)
    
        original_origin = filenameT1.GetOrigin()
        original_direction = filenameT1.GetDirection()
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(imgOrg.GetDirection())
        resampler.SetOutputOrigin(imgOrg.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)
    
        resampled_filenameT1 = resampler.Execute(filenameT1)
        resampled_filenameT1.SetSpacing(new_spacing)
    
        # Save resampled image
        #sitk.WriteImage(resampled_filenameT1, os.path.join(datapath_output, f'{subject_id}-resample.nii.gz'))
    
        
        #print('Histogram martching ...')
        matched_data =hist_match(resampled_filenameT1, reference_data)
        matched_data_array = sitk.GetArrayFromImage(matched_data)
        
        matRecon, matRecon_ori, matLabel = cropCubic_first(mrimg,new_spacing,new_size,original_spacing,original_size,original_origin, original_direction,matched_data_array,subject_id,device,model)
        
        result_nii = sitk.GetImageFromArray(matRecon_ori)
        
        ref_nii = imgOrg
        result_nii.SetOrigin(ref_nii.GetOrigin())
        result_nii.SetDirection(ref_nii.GetDirection())
        result_nii.SetSpacing(ref_nii.GetSpacing())
        pixelID = result_nii.GetPixelID()
        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(pixelID)
        result_nii = caster.Execute(result_nii)
        result_nii = sitk.Cast(result_nii, sitk.sitkUInt16)  #Uint16 is short type;
        
        sitk.WriteImage(result_nii, '/{}/{}-brain-enhanced.nii.gz'.format(datapath_output, subject_id))
        
        result_nii = sitk.Cast(result_nii, sitk.sitkFloat32)
        matched_matRecon =hist_match(result_nii,filenameT1_forhist)      
        matched_matRecon_array = sitk.GetArrayFromImage(matched_matRecon)
        skull_maskimg = 1 - maskimg
        skull_img = mrimg_ori*skull_maskimg
        matRecon_withSkull = skull_img+matched_matRecon_array
        matRecon_withSkull = sitk.GetImageFromArray(matRecon_withSkull)
        result_nii_withSkull = matRecon_withSkull
        result_nii_withSkull.SetOrigin(ref_nii.GetOrigin())
        result_nii_withSkull.SetDirection(ref_nii.GetDirection())
        result_nii_withSkull.SetSpacing(ref_nii.GetSpacing())
        pixelID = result_nii.GetPixelID()
        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(pixelID)
        result_nii_withSkull = caster.Execute(result_nii_withSkull)
        result_nii_withSkull = sitk.Cast(result_nii_withSkull, sitk.sitkUInt16)  #Uint16 is short type;z
        sitk.WriteImage(result_nii_withSkull, '/{}/{}-enhanced.nii.gz'.format(datapath_output, subject_id))
        
        # Calculate TCT
        matched_data =hist_match(imgOrg, reference_data)
        mrimg_ori = sitk.GetArrayFromImage(matched_data)
        ori_gm_values = mrimg_ori[matLabel == 2]
        ori_wm_values = mrimg_ori[matLabel == 3]
        recon_gm_values = matRecon_ori[matLabel == 2]
        recon_wm_values = matRecon_ori[matLabel == 3]
        
        
        mean_ori_gm = np.mean(ori_gm_values)      
        mean_ori_wm = np.mean(ori_wm_values)
        std_ori_gm = np.std(ori_gm_values)
        std_ori_wm = np.std(ori_wm_values)
        
        mean_recon_gm = np.mean(recon_gm_values)
        mean_recon_wm = np.mean(recon_wm_values)
        std_recon_gm = np.std(recon_gm_values)
        std_recon_wm = np.std(recon_wm_values)

        ori_tct_score = calculate_tct(mean_ori_gm, mean_ori_wm, std_ori_gm, std_ori_wm)
        recon_tct_score = calculate_tct(mean_recon_gm, mean_recon_wm, std_recon_gm, std_recon_wm)
        qi = ori_tct_score / recon_tct_score

        print(f"QI of the Original Image: {qi:.4f}")

        ori_subject_id = subject_id[:-9]
        file_path = f"{datapath_output}/{ori_subject_id}-QI.txt"
        with open(file_path, "w") as f:
            f.write(f"Quality Index (QI) of the Original Image: {qi:.4f}\n")
            f.write("Quality index [0,  ], where a higher value indicates better quality (0 being the worst).\n")

        
        # ori_tct_score = calculate_tct(mean_ori_gm, mean_ori_wm, std_ori_gm, std_ori_wm)
        # recon_tct_score = calculate_tct(mean_recon_gm, mean_recon_wm, std_recon_gm, std_recon_wm)
        # print(f"Tissue Contrast T-score (TCT) of the Original Image: {ori_tct_score:.4f}")
        # print(f"Tissue Contrast T-score (TCT) of the Enhanced Image: {recon_tct_score:.4f}")
        #
        # ori_subject_id = subject_id[:-9]
        # file_path = f"{datapath_output}/{ori_subject_id}-TCT.txt"
        # with open(file_path, "w") as f:
        #     f.write(f"Tissue Contrast T-score (TCT) of the Original Image: {ori_tct_score:.4f}\n")
        #     f.write(f"Tissue Contrast T-score (TCT) of the Enhanced Image: {recon_tct_score:.4f}")
        #
        # print('Data processing completed!')
        
#if __name__ == '__main__':
#    main()