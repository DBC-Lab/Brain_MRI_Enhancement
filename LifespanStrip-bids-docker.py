import os
import torch
import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom
from scipy.ndimage import label as ndimage_label
from scipy.ndimage import sum as ndi_sum
from monai.inferers import sliding_window_inference
import math
from networks.net import NET
from DUNet3D_T1w import DenseUNet3d_T1w
from DUNet3D_T2w import DenseUNet3d_T2w
import argparse
import SimpleITK as sitk
from bids import *
from utils.bids import *
from utils.utils import *
import torch.nn.functional as F
import json
import warnings
from BME_X.BME_X_enhanced import BMEX
from colorama import Fore, Style, init
# from BME_X.BIDS_data import parse_bids_for_age_months

warnings.filterwarnings("ignore", message="Analyze file and it's deprecated")
warnings.filterwarnings("ignore", message="torch.meshgrid")
warnings.filterwarnings("ignore", message="Implicit dimension choice for softmax")
warnings.filterwarnings("ignore", message="align_corners=False since 1.3.0")

parser1 = argparse.ArgumentParser(description='LifespanStrip pipeline')
parser1.add_argument('--pretrained_dir', default='./Model/', type=str, help='pretrained checkpoint directory')
parser1.add_argument('--bids_root', type=str, help='BIDS dataset directory')
parser1.add_argument('--subject_id', default='', type=str, help='subject_id')
parser1.add_argument('--session_id', default='', type=str, help='session_id')
parser1.add_argument('--bids_filter_file', type=str, help='Path to BIDS filter file (JSON)')
parser1.add_argument('--pretrained_model_name', default='T1w-model.pt', type=str, help='pretrained model name')
parser1.add_argument('--saved_checkpoint', default='ckpt', type=str,
                     help='Supports torchscript or ckpt pretrained checkpoint type')
parser1.add_argument('--infer_overlap', default=0.8, type=float, help=argparse.SUPPRESS)
parser1.add_argument('--in_channels', default=1, type=int, help=argparse.SUPPRESS)
parser1.add_argument('--out_channels', default=4, type=int, help=argparse.SUPPRESS)
parser1.add_argument('--roi_x', default=64, type=int, help=argparse.SUPPRESS)
parser1.add_argument('--roi_y', default=64, type=int, help=argparse.SUPPRESS)
parser1.add_argument('--roi_z', default=64, type=int, help=argparse.SUPPRESS)
parser1.add_argument('--workers', default=1, type=int, help=argparse.SUPPRESS)
parser1.add_argument('--N4', default='True', type=str, help='apply N4')


def read_bids_filter(file_path):
    """Read and parse the BIDS filter file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def main():
    args = parser1.parse_args()
    args.test_mode = True
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth_T1w = os.path.join(args.pretrained_dir, model_name)

    model1 = DenseUNet3d_T1w()
    model1_dict = torch.load(pretrained_pth_T1w, map_location=('cpu'))
    model1.load_state_dict(model1_dict['state_dict'])

    model1.eval()
    model1.to(device)

    pretrained_pth_T2w = os.path.join(args.pretrained_dir, 'T2w-model.pt')
    model2 = DenseUNet3d_T2w()
    model2_dict = torch.load(pretrained_pth_T2w, map_location=('cpu'))
    model2.load_state_dict(model2_dict['state_dict'])

    model2.eval()
    model2.to(device)

    with torch.no_grad():
        # Initialize BIDSLayout with validation
        #data_base = '/app/data/'
        data_base = './'
        #layout = BIDSLayout(data_base + args.bids_root, validate=False)
        layout = BIDSLayout('/home/limeiw/LifespanStrip_BMEX/'+args.bids_root, validate=False)

        # Filter by subject_id and session_id if both are provided
        if args.subject_id and args.session_id:
            # Both subject_id and session_id are provided
            t1w_files = layout.get(subject=args.subject_id, session=args.session_id, extension=['nii.gz', 'nii'],
                                   suffix=['T1w', 'T2w'])
            print(f"Found {len(t1w_files)} files for subject {args.subject_id} and session {args.session_id}")
        elif args.subject_id:
            # Only subject_id is provided
            t1w_files = layout.get(subject=args.subject_id, extension=['nii.gz', 'nii'], suffix=['T1w', 'T2w'])
            print(f"Found {len(t1w_files)} files for subject {args.subject_id}")
        elif args.session_id:
            # Only session_id is provided
            t1w_files = layout.get(session=args.session_id, extension=['nii.gz', 'nii'], suffix=['T1w', 'T2w'])
            print(f"Found {len(t1w_files)} files for session {args.session_id}")
        else:
            # If neither subject_id nor session_id is provided
            t1w_files = layout.get(extension=['nii.gz', 'nii'], suffix=['T1w', 'T2w'])
            print(f"Found {len(t1w_files)} files in the dataset")

        # t1w_files = layout.get(extension='nii', suffix='T1w', **filters)
        # print(t1w_files)
        if not t1w_files:
            raise ValueError("No T1w files found matching the specified filters.")

        for t1w_file in t1w_files:
            file_path = t1w_file.path

            file_name = os.path.basename(file_path)

            # subject_id 和 session_id
            subject_part = file_name.split("_")[0]
            session_part = file_name.split("_")[1]

            subject_id = subject_part.replace("sub-", "")
            session_id = session_part.replace("ses-", "")

            print(f"Subject ID: {subject_id}")
            print(f"Session ID: {session_id}")
            print(f"Processing file: {file_path}")

            # Read and preprocess MRI image
            img_name = os.path.splitext(os.path.splitext(file_path)[0])[0]

            # reorient
            print(f"Reorient to RAI direction")
            reoriented_base = f"{img_name}-reorient"
            # swapped_base = f"{img_name}-reorient"
            T1w_img_reorient = reorient_to_std(file_path, reoriented_base)
            # T1w_img_reorient = swap_dimensions(T1w_img_std, swapped_base)

            # print new direction
            T1w_img_reorient = sitk.ReadImage(T1w_img_reorient)
            size, origin, spacing, direction = T1w_img_reorient.GetSize(), T1w_img_reorient.GetOrigin(), T1w_img_reorient.GetSpacing(), T1w_img_reorient.GetDirection()
            # print('Reorientation done')

            # Apply N4 bias correction if specified
            if args.N4 == 'True':
                print(f"Bias correction (N4)")
                T1w_img_reorient = sitk.Cast(T1w_img_reorient, sitk.sitkFloat32)
                corrector = sitk.N4BiasFieldCorrectionImageFilter()
                corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
                T1w_img_reorient = corrector.Execute(T1w_img_reorient)
                # print('N4 done')
                save_dir = img_name + '-reorient-n4.nii.gz'
                out = T1w_img_reorient
                sitk.WriteImage(out, save_dir)
            else:
                print('Skip N4')

            # downsample to resolution 1.6
            print(f"Downsample to 1.6*1.6*1.6")
            old_size = np.array(T1w_img_reorient.GetSize())  # [x, y, z]
            old_spacing = np.array(T1w_img_reorient.GetSpacing())  # [x, y, z]
            new_spacing = np.array([1.6, 1.6, 1.6])
            new_size = (old_size * old_spacing / new_spacing).astype(int)  # 保持物理尺寸不变

            old_origin = np.array(T1w_img_reorient.GetOrigin())
            old_center = old_origin + (old_size * old_spacing) / 2.0
            new_origin = old_center - (new_size * new_spacing) / 2.0

            resampler = sitk.ResampleImageFilter()
            resampler.SetSize([int(new_size[0]), int(new_size[1]), int(new_size[2])])
            resampler.SetOutputSpacing([1.6, 1.6, 1.6])
            resampler.SetOutputOrigin(new_origin.tolist())  # 这里要用 list
            resampler.SetOutputDirection(T1w_img_reorient.GetDirection())
            resampler.SetInterpolator(sitk.sitkLinear)
            T1w_img_reorient_downsample = resampler.Execute(T1w_img_reorient)

            # save downsampled MRI image
            # save_dir = img_name + '-reorient-downsample.nii.gz'
            # sitk.WriteImage(T1w_img_reorient_downsample, save_dir)

            # Load and histogram-match the template
            print(f"Histogram matching")
            template = sitk.ReadImage('./Template/template.hdr')
            T1w_img_reorient_downsample = sitk.Cast(T1w_img_reorient_downsample, sitk.sitkFloat32)
            template = sitk.Cast(template, sitk.sitkFloat32)
            matcher = sitk.HistogramMatchingImageFilter()
            matcher.SetNumberOfHistogramLevels(1024)
            matcher.SetNumberOfMatchPoints(7)
            matcher.ThresholdAtMeanIntensityOn()
            T1w_img_reorient_downsample_hm = matcher.Execute(T1w_img_reorient_downsample, template)

            # save hm MRI image
            # save_dir = img_name + '-reorient-downsample-hm.nii.gz'
            # sitk.WriteImage(T1w_img_reorient_downsample_hm, save_dir)

            print(f"Skull stripping")
            T1w_img_reorient_downsample_hm = sitk.GetArrayFromImage(T1w_img_reorient_downsample_hm)
            T1w_img_reorient_downsample_hm = torch.tensor(T1w_img_reorient_downsample_hm).float()
            T1w_img_reorient_downsample_hm = T1w_img_reorient_downsample_hm.unsqueeze(dim=0)
            T1w_img_reorient_downsample_hm = T1w_img_reorient_downsample_hm.unsqueeze(dim=0)

            T1w_img_reorient_downsample_hm = T1w_img_reorient_downsample_hm.to(device)

            # T1w_img = (T1w_img - torch.min(T1w_img)) / (torch.max(T1w_img) - torch.min(T1w_img))
            T1w_img_reorient_downsample_hm = T1w_img_reorient_downsample_hm / 10000.00

            if 'T1w' in img_name:
                logits = sliding_window_inference(T1w_img_reorient_downsample_hm, (args.roi_x, args.roi_y, args.roi_z),
                                                  6, model1,
                                                  overlap=args.infer_overlap)
                promap = logits[:, :2, :, :, :]
                suffix = 'T1w'

            if 'T2w' in img_name:
                logits = sliding_window_inference(T1w_img_reorient_downsample_hm, (args.roi_x, args.roi_y, args.roi_z),
                                                  6, model2,
                                                  overlap=args.infer_overlap)
                promap = logits[:, :2, :, :, :]
                promap[:, 1, :, :, :] = logits[:, 2, :, :, :] + logits[:, 3, :, :, :]
                promap[:, 0, :, :, :] = 1 - promap[:, 1, :, :, :]
                suffix = 'T2w'

            print(f"Upsample to original space")
            promap_upsampled = F.interpolate(promap, size=[size[2], size[1], size[0]], mode='trilinear',
                                             align_corners=True)
            promap_upsampled = promap_upsampled.cpu().numpy()
            pre = np.argmax(promap_upsampled, axis=1).astype(np.uint8)

            # Apply binary opening
            pre_b = sitk.GetImageFromArray(pre[0, :, :, :])
            opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
            opening_filter.SetKernelRadius(3)
            pre_b_o = opening_filter.Execute(pre_b)

            # Extract largest connected component
            opened_image = sitk.GetArrayFromImage(pre_b_o)
            labeled_array, _ = ndimage_label(opened_image)
            sizes = ndi_sum(opened_image, labeled_array, range(labeled_array.max() + 1))
            mask = sizes < max(sizes)
            labeled_array[mask[labeled_array]] = 0
            labeled_array, _ = ndimage_label(labeled_array)

            # Fill hole
            # mask = sitk.GetArrayFromImage(labeled_array)
            labeled_array_mask = fill_holes(labeled_array, area_threshold=1)

            print(f"Save brainmask")
            s_path = img_name + '-reorient-brainmask.nii.gz'
            out = sitk.GetImageFromArray(labeled_array_mask)
            out.SetOrigin(origin)
            out.SetSpacing(spacing)
            out.SetDirection(direction)
            sitk.WriteImage(out, s_path)

            T1w_img = sitk.ReadImage(t1w_file.path)
            ori_size, ori_origin, ori_spacing, ori_direction = T1w_img.GetSize(), T1w_img.GetOrigin(), T1w_img.GetSpacing(), T1w_img.GetDirection()

            ## BME-X: https://doi.org/10.1038/s41551-024-01283-7
            # Input: T1w_img_reorient, labeled_array_mask
            age_months = parse_bids_for_age_months(args.bids_root, subject_id, session_id)
            if age_months is None or age_months == 'None':
                init(autoreset=True)
                # print('Skipping BME-X! Age information is required!')
                print(Fore.RED + Style.BRIGHT + "Skipping BME-X! Age information is required!")
            else:
                print('age_in_month', age_months)
                file_path_for_BMEX = os.path.dirname(file_path)
                print('file_path_for_BMEX', file_path_for_BMEX)
                BMEX(input_path=file_path_for_BMEX, output_path=file_path_for_BMEX, age_number=age_months,
                     suffix=suffix, device=device)

                # Save enhanced image
                enhanced_path = img_name + '-reorient-enhanced.nii.gz'
                enhanced_img = sitk.ReadImage(enhanced_path)
                enhanced_img = sitk.GetArrayFromImage(enhanced_img)
                enhanced_img_flipped = np.flip(enhanced_img, axis=1)
                enhanced_img_flipped_flipped = np.flip(enhanced_img_flipped, axis=2)
                out = sitk.GetImageFromArray(enhanced_img_flipped_flipped)
                save_dir = img_name + '-enhanced.nii.gz'
                out.SetOrigin(ori_origin)
                out.SetSpacing(ori_spacing)
                out.SetDirection(ori_direction)
                sitk.WriteImage(out, save_dir)

                brain_enhanced_path = img_name + '-reorient-brain-enhanced.nii.gz'
                brain_enhanced_img = sitk.ReadImage(brain_enhanced_path)
                brain_enhanced_img = sitk.GetArrayFromImage(brain_enhanced_img)
                brain_enhanced_img_flipped = np.flip(brain_enhanced_img, axis=1)
                brain_enhanced_img_flipped_flipped = np.flip(brain_enhanced_img_flipped, axis=2)
                out = sitk.GetImageFromArray(brain_enhanced_img_flipped_flipped)
                save_dir = img_name + '-brain-enhanced.nii.gz'
                out.SetOrigin(ori_origin)
                out.SetSpacing(ori_spacing)
                out.SetDirection(ori_direction)
                sitk.WriteImage(out, save_dir)

                os.remove(img_name + '-reorient-enhanced.nii.gz')
                os.remove(img_name + '-reorient-brain-enhanced.nii.gz')

            # Save the restored image
            T1w_img_mask = labeled_array_mask
            T1w_img_mask_flipped = np.flip(T1w_img_mask, axis=1)
            T1w_img_mask_flipped_flipped = np.flip(T1w_img_mask_flipped, axis=2)
            save_dir = img_name + '-brainmask.nii.gz'
            out = sitk.GetImageFromArray(T1w_img_mask_flipped_flipped)
            out.SetOrigin(ori_origin)
            out.SetSpacing(ori_spacing)
            out.SetDirection(ori_direction)
            sitk.WriteImage(out, save_dir)

            os.remove(img_name + '-reorient-n4.nii.gz')
            os.remove(img_name + '-reorient-brainmask.nii.gz')
            os.remove(img_name + '-reorient.nii.gz')


if __name__ == '__main__':
    main()
