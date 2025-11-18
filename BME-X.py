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
from DUNet3D import DenseUNet3d
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
import shutil
import re

init(autoreset=True)
#from BME_X.BIDS_data import parse_bids_for_age_months

warnings.filterwarnings("ignore", message="Analyze file and it's deprecated")
warnings.filterwarnings("ignore", message="torch.meshgrid")
warnings.filterwarnings("ignore", message="Implicit dimension choice for softmax")
warnings.filterwarnings("ignore", message="align_corners=False since 1.3.0")

parser1 = argparse.ArgumentParser(
    description=(
        "This is for skull stripping (LifespanStrip) and brain MRIs enhancement (BME-X),"
        "by Developing Brain Computing Lab, University of North Carolina at Chapel Hill\n"
        "Version: v1.0.5\n"
        "Authors: Li Wang, Yue Sun, Limei Wang\n"
        "\n"
        "If you find it useful, please kindly cite:\n"
        "  (1) Sun, Y., Wang, L., Li, G. et al. A foundation model for enhancing magnetic resonance images and downstream "
        "segmentation, registration and diagnostic tasks. Nat. Biomed. Eng 9, 521–538 (2025). https://doi.org/10.1038/s41551-024-01283-7\n"
        "  (2) Wang, L., Sun, Y., Seidlitz, J. et al. A lifespan-generalizable skull-stripping model for magnetic resonance images "
        "that leverages prior knowledge from brain atlases. Nat. Biomed. Eng 9, 700–715 (2025). https://doi.org/10.1038/s41551-024-01337-w\n"
        "\n"
        "Contacts:\n"
        "  - li_wang@med.unc.edu\n"
        "  - yuesun@med.unc.edu\n"
        "  - limeiw@med.unc.edu\n"
        "\n"
        "Code Repository:\n"
        "  - https://github.com/DBC-Lab/Brain_MRI_Enhancement\n"
        "  - https://github.com/DBC-Lab/Atlases-empowered_Lifespan_Skull_Stripping\n"
        "\n"
        "Copyright (C) 2025 DBC Lab. All rights reserved.\n"
        "------------------------------------------\n"
        "\n"
        "Input:\n"
    ),
    
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=(
        "Outputs (BIDS derivatives style):\n"
        "\n"
        "*_<T1w|T2w>.nii.gz\n"
        "    Unprocessed copy placed in output/anat.\n"
        "    Sidecar JSON includes: Sources, SpatialReference, and SkullStripped:false.\n"
        "\n"
        "*_desc-preproc_<T1w|T2w>.nii.gz\n"
        "    Enhanced image WITHOUT skull stripping.\n"
        "    Sidecar JSON includes: Sources, SpatialReference, SkullStripped:false, and QI {Quality index [0,  ], where a higher value indicates better quality (0 being the worst)}.\n"
        "\n"
        "*_desc-enhanced_<T1w|T2w>.nii.gz\n"
        "    Enhanced image WITH skull stripping.\n"
        "    Sidecar JSON includes: Sources, SpatialReference, SkullStripped:true, and QI {Quality index [0,  ], where a higher value indicates better quality (0 being the worst)}.\n"
        "\n"
        "*_desc-brain_mask.nii.gz\n"
        "    Binary brain mask (1=brain, 0=non-brain). JSON: SkullStripped:true.\n"
        "    Sidecar JSON includes: Sources, SpatialReference, Type: Brain, and SkullStripped:true.\n"
        "\n"
    )
)
parser1.add_argument('--pretrained_dir', default='/Model/', type=str, help=argparse.SUPPRESS)
parser1.add_argument('--bids_root', type=str, help='BIDS dataset directory')
parser1.add_argument('--subject_id', default='', type=str, help='subject_id')
parser1.add_argument('--session_id', default='', type=str, help='session_id')
parser1.add_argument('--bids_filter_file', type=str, help=argparse.SUPPRESS)
parser1.add_argument('--pretrained_model_name', default='T1T2w-model.pt', type=str, help=argparse.SUPPRESS)
parser1.add_argument('--saved_checkpoint', default='ckpt', type=str, help=argparse.SUPPRESS)
parser1.add_argument('--infer_overlap', default=0.85, type=float, help=argparse.SUPPRESS)
parser1.add_argument('--in_channels', default=1, type=int, help=argparse.SUPPRESS)
parser1.add_argument('--out_channels', default=4, type=int, help=argparse.SUPPRESS)
parser1.add_argument('--roi_x', default=64, type=int, help=argparse.SUPPRESS)
parser1.add_argument('--roi_y', default=64, type=int, help=argparse.SUPPRESS)
parser1.add_argument('--roi_z', default=64, type=int, help=argparse.SUPPRESS)
parser1.add_argument('--workers', default=1, type=int, help=argparse.SUPPRESS)
parser1.add_argument('--N4', default='True', type=str, help=argparse.SUPPRESS)
parser1.add_argument('--data_base', default='', type=str, help='data_base')
parser1.add_argument('--output_dir', default='/output', type=str, help='output_dir')

# --- sidecar helpers: write JSON next to a .nii.gz ---
def _nifti_to_sidecar_path(nifti_path: str) -> str:
    # *.nii.gz  →  *.json
    base = os.path.splitext(os.path.splitext(nifti_path)[0])[0]
    return base + ".json"

def _to_bids_raw(path_on_disk: str, bids_root_dir: str) -> str:
    # make "bids:raw:/sub-xx/ses-xx/..." from absolute path under bids_root_dir
    import os
    rel = os.path.relpath(path_on_disk, start=bids_root_dir)
    return "bids:raw:" + rel.replace(os.sep, "/")

def write_sidecar_json(nifti_path: str, sources_paths: list, spatial_ref_path: str,
                       bids_root_dir: str, skull_stripped: bool, qi_value: float | None = None, type_field: str | None = None):
    """Write the requested fields into a sidecar JSON next to nifti_path; optionally attach QI."""
    sidecar = _nifti_to_sidecar_path(nifti_path)
    payload = {
        "Sources": [_to_bids_raw(p, bids_root_dir) for p in sources_paths],
        "SpatialReference": _to_bids_raw(spatial_ref_path, bids_root_dir),
        "SkullStripped": bool(skull_stripped),
    }
    if qi_value is not None:
        payload["QI"] = {
            "value": float(qi_value),
            "description": "Quality index [0, 1], where a higher value indicates better quality (0 being the worst)."
        }
    if type_field is not None:         
        payload["Type"] = type_field
        
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        
def read_bids_filter(file_path):
    """Read and parse the BIDS filter file."""
    with open(file_path, 'r') as f:
        return json.load(f)

# --- QI helper: try to load a numeric QI from a sibling *-QI.txt ---
def _load_qi_value(out_dir: str, stem: str):
    """
    Try reading {stem}-QI.txt in out_dir; return float or None if unavailable.
    """
    qi_txt = os.path.join(out_dir, f"{stem}-QI.txt")
    if os.path.isfile(qi_txt):
        try:
            with open(qi_txt, "r", encoding="utf-8") as f:
                import re
                text = f.read()
                m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
                return float(m.group(0)) if m else None
        except Exception:
            return None
    return None

def write_dataset_description_json(bids_out_root: str):
    """
    Write a BIDS-compliant dataset_description.json at the top-level of the output BIDS dir.
    Safe to call multiple times; it won't overwrite if exists.
    """
    dd_path = os.path.join(bids_out_root, "dataset_description.json")
    if os.path.exists(dd_path):
        return

    payload = {
        "Name": "BME-X Outputs",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "BME-X",
                "Version": "v1.0.5",
                "Container": {
                    "Type": "docker",
                    "Tag": "yuesun814/bme-x:v1.0.5"
                }
            }
        ]
    }

    os.makedirs(bids_out_root, exist_ok=True)
    with open(dd_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        
def write_descriptions_tsv(bids_out_root: str):
    """
    Optional: Write a minimal descriptions.tsv describing desc terms.
    """
    tsv_path = os.path.join(bids_out_root, "descriptions.tsv")
    if os.path.exists(tsv_path):
        return
    rows = [
        ["desc", "meaning"],
        ["preproc", "Enhanced image without skull stripping (derived preprocessing)."],
        ["enhanced", "Enhanced image with skull stripping applied."]
    ]
    with open(tsv_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")
        
def main():
    args = parser1.parse_args()
    args.test_mode = True
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(args.pretrained_dir, model_name)

    model1 = DenseUNet3d()
    model1_dict = torch.load(pretrained_pth, map_location=('cpu'))
    model1.load_state_dict(model1_dict['state_dict'])

    model1.eval()
    model1.to(device)
    
    write_dataset_description_json(args.output_dir)
    #write_descriptions_tsv(args.output_dir) 

    with torch.no_grad():
        # ---- Construct the BIDS root path (supports empty --data_base and absolute --bids_root) ----
        data_base = os.path.abspath(args.data_base) if args.data_base else ''
        bids_root_arg = args.bids_root or ''
        if os.path.isabs(bids_root_arg):
            layout_root = bids_root_arg
        else:
            layout_root = os.path.abspath(os.path.join(data_base, bids_root_arg))

        if not os.path.isdir(layout_root):
            raise FileNotFoundError(f"BIDS root not found: {layout_root}")

        layout = BIDSLayout(layout_root, validate=False)
    
        if getattr(args, "output_dir", None):
            args.output_dir = os.path.abspath(args.output_dir)
            os.makedirs(args.output_dir, exist_ok=True)
            
        # Filter by subject_id and session_id if both are provided
        if args.subject_id and args.session_id:
            # Both subject_id and session_id are provided
            t1w_files = layout.get(subject=args.subject_id, session=args.session_id, extension=['nii.gz', 'nii'], suffix=['T1w', 'T2w'])
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

        #t1w_files = layout.get(extension='nii', suffix='T1w', **filters)
        #print(t1w_files)
        if not t1w_files:
            raise ValueError("No T1w files found matching the specified filters.")

        for t1w_file in t1w_files:
            file_path = t1w_file.path
            
            file_name = os.path.basename(file_path)

            #subject_id 和 session_id
            subject_part = file_name.split("_")[0]
            session_part = file_name.split("_")[1]

            subject_id = subject_part.replace("sub-", "")
            session_id = session_part.replace("ses-", "")
            
            suffix = 'T1w' if ('T1w' in file_name or 'T1w' in file_path) else 'T2w'

            print(f"Subject ID: {subject_id}")
            print(f"Session ID: {session_id}")
            print(f"Processing file: {file_path}")

            # Read and preprocess MRI image
            img_name = os.path.splitext(os.path.splitext(file_path)[0])[0]
            stem = os.path.basename(img_name)
            stem_no_suffix = re.sub(r'_(T1w|T2w)$', '', stem)
            
            # --- output dirs ---
            out_root = args.output_dir if getattr(args, "output_dir", None) else os.path.dirname(file_path)
            out_dir = os.path.join(out_root, f"sub-{subject_id}", f"ses-{session_id}", "anat")
            os.makedirs(out_dir, exist_ok=True)
            print("Output dir (this run):", out_dir)

            #dst_path = os.path.join(out_dir, f"sub-{subject_id}_ses-{session_id}_{suffix}.nii.gz")
            dst_path = os.path.join(out_dir, os.path.basename(file_path))
            shutil.copy2(file_path, dst_path)
            
            img_name = os.path.splitext(os.path.splitext(file_path)[0])[0]
            stem = os.path.basename(img_name)
            qi_val = _load_qi_value(out_dir, stem)

            write_sidecar_json(
                nifti_path=dst_path,
                sources_paths=[file_path],
                spatial_ref_path=file_path,
                bids_root_dir=layout_root,   
                skull_stripped=False,
                qi_value=qi_val
            )

            # reorient
            print(f"Reorient to RAI direction")
            #reoriented_base = f"{img_name}-reorient"
            reoriented_base = os.path.join(out_dir, f"{stem}-reorient")
            #swapped_base = f"{img_name}-reorient"
            T1w_img_reorient = reorient_to_std(file_path, reoriented_base)
            #T1w_img_reorient = swap_dimensions(T1w_img_std, swapped_base)


            # print new direction
            T1w_img_reorient = sitk.ReadImage(T1w_img_reorient)
            size, origin, spacing, direction = T1w_img_reorient.GetSize(), T1w_img_reorient.GetOrigin(), T1w_img_reorient.GetSpacing(), T1w_img_reorient.GetDirection()
            #print('Reorientation done')
            
            #rescale the image intensity to 0~1000
            arr = sitk.GetArrayFromImage(T1w_img_reorient).astype(np.float32)
            mask = np.isfinite(arr) & (arr != 0)
            vals = arr[mask] if np.any(mask) else arr[np.isfinite(arr)]
            
            if vals.size == 0:
                p1, p99 = 0.0, 1.0
            else:
                p1, p99 = np.nanpercentile(vals, [0.001, 99.999])
                if not np.isfinite(p1) or not np.isfinite(p99) or p1 >= p99:
                    finite = arr[np.isfinite(arr)]
                    if finite.size > 0:
                        p1, p99 = np.percentile(finite, [0.001, 99.999])
                    else:
                        p1, p99 = 0.0, 1.0
                 
            img_f = sitk.Cast(T1w_img_reorient, sitk.sitkFloat32)
            T1w_img_reorient = sitk.IntensityWindowing(img_f, p1, p99, 0.0, 1000.0)

            # Apply N4 bias correction if specified
            if args.N4 == 'True':
                print(f"Bias correction (N4)")
                T1w_img_reorient = sitk.Cast(T1w_img_reorient, sitk.sitkFloat32)
                corrector = sitk.N4BiasFieldCorrectionImageFilter()
                corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
                T1w_img_reorient = corrector.Execute(T1w_img_reorient)
                #print('N4 done')
                save_dir = os.path.join(out_dir, f"{stem}-reorient-n4.nii.gz")
                out = T1w_img_reorient
                sitk.WriteImage(out, save_dir)
            else:
                print('Skip N4')



            # downsample to resolution 1.6
            print(f"Downsample to 1.6*1.6*1.6")
            old_size = np.array(T1w_img_reorient.GetSize())  # [x, y, z]
            old_spacing = np.array(T1w_img_reorient.GetSpacing())  # [x, y, z]
            new_spacing = np.array([1.6, 1.6, 1.6])
            new_size = (old_size * old_spacing / new_spacing).astype(int)  

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
            #save_dir = img_name + '-reorient-downsample.nii.gz'
            #sitk.WriteImage(T1w_img_reorient_downsample, save_dir)

            # Load and histogram-match the template
            print(f"Histogram matching")
            template = sitk.ReadImage('/Template/template.hdr')
            T1w_img_reorient_downsample = sitk.Cast(T1w_img_reorient_downsample, sitk.sitkFloat32)
            template = sitk.Cast(template, sitk.sitkFloat32)
            matcher = sitk.HistogramMatchingImageFilter()
            matcher.SetNumberOfHistogramLevels(1024)
            matcher.SetNumberOfMatchPoints(7)
            matcher.ThresholdAtMeanIntensityOn()
            T1w_img_reorient_downsample_hm = matcher.Execute(T1w_img_reorient_downsample, template)

            # save hm MRI image
            #save_dir = img_name + '-reorient-downsample-hm.nii.gz'
            #sitk.WriteImage(T1w_img_reorient_downsample_hm, save_dir)

            print(f"Skull stripping")
            T1w_img_reorient_downsample_hm = sitk.GetArrayFromImage(T1w_img_reorient_downsample_hm)
            T1w_img_reorient_downsample_hm = torch.tensor(T1w_img_reorient_downsample_hm).float()
            T1w_img_reorient_downsample_hm = T1w_img_reorient_downsample_hm.unsqueeze(dim=0)
            T1w_img_reorient_downsample_hm = T1w_img_reorient_downsample_hm.unsqueeze(dim=0)

            T1w_img_reorient_downsample_hm = T1w_img_reorient_downsample_hm.to(device)

            # T1w_img = (T1w_img - torch.min(T1w_img)) / (torch.max(T1w_img) - torch.min(T1w_img))
            T1w_img_reorient_downsample_hm = T1w_img_reorient_downsample_hm / 10000.00


            logits = sliding_window_inference(T1w_img_reorient_downsample_hm, (args.roi_x, args.roi_y, args.roi_z),
                                              4, model1,
                                              overlap=args.infer_overlap)
            promap = logits[:, :2, :, :, :]
            promap[:, 1, :, :, :] = logits[:, 2, :, :, :] + logits[:, 3, :, :, :]
            promap[:, 0, :, :, :] = 1 - promap[:, 1, :, :, :]

            print(f"Upsample to original space")
            promap_upsampled = F.interpolate(promap, size=[size[2], size[1], size[0]], mode='trilinear', align_corners=True)
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
            #mask = sitk.GetArrayFromImage(labeled_array)
            labeled_array_mask = fill_holes(labeled_array, area_threshold=1)

            print(f"Save brainmask")
            s_path = os.path.join(out_dir, f"{stem}-reorient-brainmask.nii.gz")
            out = sitk.GetImageFromArray(labeled_array_mask)
            out.SetOrigin(origin)
            out.SetSpacing(spacing)
            out.SetDirection(direction)
            sitk.WriteImage(out, s_path)

            T1w_img = sitk.ReadImage(t1w_file.path)
            ori_size, ori_origin, ori_spacing, ori_direction = T1w_img.GetSize(), T1w_img.GetOrigin(), T1w_img.GetSpacing(), T1w_img.GetDirection()

            
            ## BME-X: https://doi.org/10.1038/s41551-024-01283-7
            #Input: T1w_img_reorient, labeled_array_mask
            root = data_base+args.bids_root
            #root = args.bids_root
            age_months = parse_bids_for_age_months(root, subject_id, session_id)
            if age_months is None or age_months == 'None':
                print('-----------------------------------------------------')
                print('|                                                   |')
                print('|   Skipping BME-X! Age information is required!    |')
                print('|                                                   |')
                print('-----------------------------------------------------')
                #print(Fore.RED + Style.BRIGHT + "Skipping BME-X! Age information is required!")
            else:
                print('age_in_month', age_months)
                #combined_file_path = os.path.join(out_root, os.path.basename(out_dir))
                #file_path_for_BMEX = os.path.dirname(combined_file_path)
                
                #file_path_for_BMEX = os.path.join(out_dir, f"{stem}-reorient-enhanced.nii.gz")
                file_path_for_BMEX = out_dir
                
                #print("combined_file_path:", combined_file_path)
                print("file_path_for_BMEX:", file_path_for_BMEX)
                BMEX(input_path=file_path_for_BMEX, output_path=file_path_for_BMEX, age_number=age_months,
                     suffix=suffix, device=device)

                # Save enhanced image
                enhanced_path =  os.path.join(out_dir, f"{stem}-reorient-enhanced.nii.gz")
                enhanced_img = sitk.ReadImage(enhanced_path)
                enhanced_img = sitk.GetArrayFromImage(enhanced_img)
                enhanced_img_flipped = np.flip(enhanced_img, axis=1)
                enhanced_img_flipped_flipped = np.flip(enhanced_img_flipped, axis=2)
                # out = sitk.GetImageFromArray(enhanced_img_flipped_flipped)
                # enhanced_out = os.path.join(out_dir, f"{stem}-enhanced.nii.gz")
                # out.SetOrigin(ori_origin)
                # out.SetSpacing(ori_spacing)
                # out.SetDirection(ori_direction)
                # sitk.WriteImage(out, enhanced_out)
                
                # enhanced_out = os.path.join(out_dir, f"{stem}_desc-preproc_{suffix}.nii.gz")
                enhanced_out = os.path.join(out_dir, f"{stem_no_suffix}_desc-preproc_{suffix}.nii.gz")
                out = sitk.GetImageFromArray(enhanced_img_flipped_flipped)  
                out.SetOrigin(ori_origin); out.SetSpacing(ori_spacing); out.SetDirection(ori_direction)
                sitk.WriteImage(out, enhanced_out)

                brain_enhanced_path = os.path.join(out_dir, f"{stem}-reorient-brain-enhanced.nii.gz")
                brain_enhanced_img = sitk.ReadImage(brain_enhanced_path)
                brain_enhanced_img = sitk.GetArrayFromImage(brain_enhanced_img)
                brain_enhanced_img_flipped = np.flip(brain_enhanced_img, axis=1)
                brain_enhanced_img_flipped_flipped = np.flip(brain_enhanced_img_flipped, axis=2)
                # out = sitk.GetImageFromArray(brain_enhanced_img_flipped_flipped)
                # brain_enhanced_out = os.path.join(out_dir, f"{stem}-brain-enhanced.nii.gz")
                # out.SetOrigin(ori_origin)
                # out.SetSpacing(ori_spacing)
                # out.SetDirection(ori_direction)
                # sitk.WriteImage(out, brain_enhanced_out)
                
                # brain_enhanced_out = os.path.join(out_dir, f"{stem}_desc-enhanced_{suffix}.nii.gz")
                brain_enhanced_out = os.path.join(out_dir, f"{stem_no_suffix}_desc-enhanced_{suffix}.nii.gz")
                out = sitk.GetImageFromArray(brain_enhanced_img_flipped_flipped)
                out.SetOrigin(ori_origin); out.SetSpacing(ori_spacing); out.SetDirection(ori_direction)
                sitk.WriteImage(out, brain_enhanced_out)
                
                # sidecar（JSON）
                qi_val_enh = _load_qi_value(out_dir, stem)
                write_sidecar_json(
                    nifti_path=brain_enhanced_out,
                    sources_paths=[t1w_file.path],
                    spatial_ref_path=t1w_file.path,
                    bids_root_dir=layout_root,
                    skull_stripped=True,
                    qi_value=qi_val_enh 
                )
                write_sidecar_json(
                    nifti_path=enhanced_out,
                    sources_paths=[t1w_file.path],
                    spatial_ref_path=t1w_file.path,
                    bids_root_dir=layout_root,
                    skull_stripped=False,
                    qi_value=qi_val_enh 
                )
                os.remove(os.path.join(out_dir, f"{stem}-reorient-enhanced.nii.gz"))
                os.remove(os.path.join(out_dir, f"{stem}-reorient-brain-enhanced.nii.gz"))



            # Save the restored image
            T1w_img_mask = labeled_array_mask
            T1w_img_mask_flipped = np.flip(T1w_img_mask, axis=1)
            T1w_img_mask_flipped_flipped = np.flip(T1w_img_mask_flipped, axis=2)
            # save_dir = os.path.join(out_dir, f"{stem}-brainmask.nii.gz")
            # out = sitk.GetImageFromArray(T1w_img_mask_flipped_flipped)
            # out.SetOrigin(ori_origin)
            # out.SetSpacing(ori_spacing)
            # out.SetDirection(ori_direction)
            # sitk.WriteImage(out, save_dir)
            
            # # save_dir ...-brainmask.nii.gz
            # write_sidecar_json(
            #     nifti_path=save_dir,
            #     sources_paths=[t1w_file.path],         
            #     spatial_ref_path=t1w_file.path,         
            #     bids_root_dir=root,                    
            #     skull_stripped=True
            # )
            
            mask_out = os.path.join(out_dir, f"{stem_no_suffix}_space-{suffix}_desc-brain_mask.nii.gz")
            out = sitk.GetImageFromArray(T1w_img_mask_flipped_flipped)
            out.SetOrigin(ori_origin); out.SetSpacing(ori_spacing); out.SetDirection(ori_direction)
            sitk.WriteImage(out, mask_out)

            write_sidecar_json(
                nifti_path=mask_out,
                sources_paths=[t1w_file.path],
                spatial_ref_path=t1w_file.path,
                bids_root_dir=layout_root,
                skull_stripped=True,
                type_field="Brain"
            )

            os.remove(os.path.join(out_dir, f"{stem}-reorient-n4.nii.gz"))
            os.remove(os.path.join(out_dir, f"{stem}-reorient-brainmask.nii.gz"))
            os.remove(os.path.join(out_dir, f"{stem}-reorient.nii.gz"))
            os.remove(os.path.join(out_dir, f"{stem}-QI.txt"))




if __name__ == '__main__':
    main()
