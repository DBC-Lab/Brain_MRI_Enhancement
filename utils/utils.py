import os
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import subprocess
from skimage.morphology import remove_small_holes
from nipype.interfaces.fsl import Reorient2Std, SwapDimensions

def reorient_to_std(input_file: str, output_base: str) -> str:
    """
    Reorient the input file to RAI orientation using np.flip and a custom affine matrix with SimpleITK.

    Args:
        input_file (str): Path to the input file.
        output_base (str): Base path (without extension) for the output file.

    Returns:
        str: Path to the generated reoriented file.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    # Load the image with SimpleITK
    img = sitk.ReadImage(input_file)

    # Get the image data as a numpy array
    img_data = sitk.GetArrayFromImage(img)

    # Flip along the Y-axis (axis 1) to reverse Posterior to Anterior direction
    img_data_flipped = np.flip(img_data, axis=1)  # Flipping Y-axis (posterior to anterior)
    img_data_flipped = np.flip(img_data_flipped, axis=2)

    # Create the reorientation matrix (RAI orientation)
    # Flip X (Left to Right), flip Y (Posterior to Anterior), keep Z as is
    reorientation_matrix = np.array([[1, 0, 0],  # Flip X axis (Left to Right)
                                     [0, 1, 0],  # Flip Y axis (Posterior to Anterior)
                                     [0, 0, 1]])  # Keep Z axis the same

    # Convert the matrix to a list of floats (SimpleITK expects this)
    matrix_list = reorientation_matrix.flatten().tolist()

    # Create a SimpleITK affine transform for the reorientation
    affine_transform = sitk.AffineTransform(3)
    affine_transform.SetMatrix(matrix_list)

    # Apply the affine transformation to the flipped data
    img_flipped = sitk.GetImageFromArray(img_data_flipped)
    img_flipped.SetSpacing(img.GetSpacing())
    img_flipped.SetOrigin(img.GetOrigin())
    img_flipped.SetDirection(affine_transform.GetMatrix())

    # Save the reoriented image
    output_file = f"{output_base}.nii.gz"
    sitk.WriteImage(img_flipped, output_file)

    if not os.path.exists(output_file):
        raise RuntimeError(f"Reoriented file not created: {output_file}")

    return output_file


def swap_dimensions(input_file: str, output_base: str) -> str:
    """
    Swap dimensions of the input file using nibabel.

    Args:
    input_file (str): Path to the input file.
    output_base (str): Base path (without extension) for the output file.

    Returns:
    str: Path to the dimension-swapped file.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    # Load the image with nibabel
    img = nib.load(input_file)

    # Swap the dimensions
    # For example, swap the x and y axes
    img_data = img.get_fdata()
    img_data = np.swapaxes(img_data, 0, 1)  # Swap x and y dimensions

    # Create the output file
    output_file = f"{output_base}.nii.gz"
    new_img = nib.Nifti1Image(img_data, img.affine)
    nib.save(new_img, output_file)

    if not os.path.exists(output_file):
        raise RuntimeError(f"Swapped dimensions file not created: {output_file}")

    return output_file



def unswap_dimensions(input_file: str, output_base: str) -> str:
    """
    Reverse the `swap_dimensions()` operation and swap the image axes back to the original order.

    Args:
        input_file (str): Path to the swapped image.
        output_base (str): Base path (without extension) for the output file.

    Returns:
        str: Path to the file after swapping the axes back.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    # Read the input image
    image = sitk.ReadImage(input_file)

    # Reverse the axis swap (if previously swapped like x <-> y)
    unswapped_image = sitk.PermuteAxes(image, (1, 0, 2))  # Reverse the previous swap: y <-> x

    # Write the output file
    output_file = f"{output_base}_unswapped.nii.gz"
    sitk.WriteImage(unswapped_image, output_file)

    if not os.path.exists(output_file):
        raise RuntimeError(f"Unswapped dimensions file not created: {output_file}")

    return output_file



def undo_reorient_to_std(input_file: str, output_base: str) -> str:
    """
    Reverse the RAI reorientation back to LPI (Left-Posterior-Inferior) using SimpleITK.

    Args:
        input_file (str): The file path of the reoriented image (RAI).
        output_base (str): Base name for the output file (without extension).

    Returns:
        str: Path to the generated undone (LPI) image.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    # Load the image with SimpleITK
    img = sitk.ReadImage(input_file)

    # Get the image data as a numpy array
    img_data = sitk.GetArrayFromImage(img)

    # Flip the Y-axis (Posterior to Anterior) to revert it back to its original orientation
    img_data_flipped = np.flip(img_data, axis=1)  # Flipping Y-axis (posterior to anterior)

    # Create the reorientation matrix for LPI (undo the RAI reorientation)
    # Flip X (Left to Right), flip Y (Posterior to Anterior), keep Z as is
    reorientation_matrix = np.array([[-1, 0, 0],  # Flip X axis (Left to Right)
                                     [0, -1, 0],  # Flip Y axis (Posterior to Anterior)
                                     [0, 0, 1]])  # Keep Z axis the same

    # Convert the matrix to a list of floats (SimpleITK expects this)
    matrix_list = reorientation_matrix.flatten().tolist()

    # Create a SimpleITK affine transform for the reorientation
    affine_transform = sitk.AffineTransform(3)
    affine_transform.SetMatrix(matrix_list)

    # Apply the affine transformation to the flipped data
    img_flipped = sitk.GetImageFromArray(img_data_flipped)
    img_flipped.SetSpacing(img.GetSpacing())
    img_flipped.SetOrigin(img.GetOrigin())
    img_flipped.SetDirection(affine_transform.GetMatrix())

    # Define the output file path
    output_file = f"{output_base}.nii.gz"

    # Save the reoriented image
    sitk.WriteImage(img_flipped, output_file)

    if not os.path.exists(output_file):
        raise RuntimeError(f"Failed to undo reorientation, file not created: {output_file}")

    return output_file


def largest_connected_component_3d(binary_image):
    # Label all connected components in the image
    labeled_array, num_features = ndimage.label(binary_image)

    # Find the sizes of the connected components
    sizes = ndimage.sum(binary_image, labeled_array, range(num_features + 1))

    # Identify the largest connected component (excluding the background)
    mask_size = sizes < max(sizes)
    remove_pixel = mask_size[labeled_array]
    labeled_array[remove_pixel] = 0

    # Label the largest connected component
    labeled_array, _ = ndimage.label(labeled_array)

    return labeled_array


def fill_holes(mask, area_threshold=100):
    """
    Fill small holes in a binary mask using skimage's remove_small_holes function.
    Args:
        mask (numpy.ndarray): 2D or 3D binary mask with holes (0-1 values).
        area_threshold (int): Minimum size of holes to fill.
    Returns:
        numpy.ndarray: Mask with holes filled.
    """
    filled_mask = remove_small_holes(mask, area_threshold=area_threshold)
    return filled_mask.astype(np.uint8)



def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0,
                            self.sum / self.count,
                            self.sum)

def distributed_all_gather(tensor_list,
                           valid_batch_size=None,
                           out_numpy=False,
                           world_size=None,
                           no_barrier=False,
                           is_valid=None):

    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g,v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out
