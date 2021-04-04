from glob import glob
import os
import copy
import numpy as np
import pyvista as pv
import SimpleITK as sitk
from shapmagn.experiments.datasets.lung.img_sampler import DataProcessing
from shapmagn.datasets.vtk_utils import read_vtk
from shapmagn.shape.shape_utils import get_scale_and_center

"""
1. take the high resoltuion (max between the insp and exp )
2.  padding at the end (for the argmin(insp, exp) image)
3. transpose the last dimension of the high dimension
"""

COPD_ID={
    "copd6":"12042G",
    "copd7":"12105E",
    "copd8":"12109M",
    "copd9":"12239Z",
    "copd10":"12829U",
    "copd1":"13216S",
    "copd2":"13528L",
    "copd3":"13671Q",
    "copd4":"13998W",
    "copd5":"17441T"
}

ID_COPD={
    "12042G":"copd6",
    "12105E":"copd7",
    "12109M":"copd8",
    "12239Z":"copd9",
    "12829U":"copd10",
    "13216S":"copd1",
    "13528L":"copd2",
    "13671Q":"copd3",
    "13998W":"copd4",
    "17441T":"copd5"
}

#in sitk coord
COPD_spacing = {"copd1": [0.625, 0.625, 2.5],
                "copd2": [0.645, 0.645, 2.5],
                "copd3": [0.652, 0.652, 2.5],
                "copd4": [0.590, 0.590, 2.5],
                "copd5": [0.647, 0.647, 2.5],
                "copd6": [0.633, 0.633, 2.5],
                "copd7": [0.625, 0.625, 2.5],
                "copd8": [0.586, 0.586, 2.5],
                "copd9": [0.664, 0.664, 2.5],
                "copd10": [0.742, 0.742, 2.5]}

# in sitk coord
COPD_shape = {"copd1": [512, 512, 121],
              "copd2": [512, 512, 102],
              "copd3": [512, 512, 126],
              "copd4": [512, 512, 126],
              "copd5": [512, 512, 131],
              "copd6": [512, 512, 119],
              "copd7": [512, 512, 112],
              "copd8": [512, 512, 115],
              "copd9": [512, 512, 116],
              "copd10":[512, 512, 135]}

def read_dirlab(file_path, shape):
    dtype = np.dtype("<i2")
    fid = open(file_path, 'rb')
    data = np.fromfile(fid, dtype)
    img_np = data.reshape(shape)
    return img_np

def save_dirlab_into_niigz(file_path, output_path, fname, is_insp=False):
    img_np = read_dirlab(file_path, np.flipud(COPD_shape[fname]))
    img_sitk = sitk.GetImageFromArray(img_np)
    # img_sitk.SetOrigin()
    img_sitk.SetSpacing(np.array(COPD_spacing[fname]))
    if is_insp:
        saving_path = os.path.join(output_path,COPD_ID[fname]+"_INSP_STD_USD_COPD.nii.gz")
    else:
        saving_path = os.path.join(output_path,COPD_ID[fname]+"_EXP_STD_USD_COPD.nii.gz")
    sitk.WriteImage(img_sitk,saving_path)
    if is_insp:
        saving_path = os.path.join(output_path,fname+"_iBHCT.nii.gz")
    else:
        saving_path = os.path.join(output_path,fname+"_eBHCT.nii.gz")
    sitk.WriteImage(img_sitk,saving_path)
    return saving_path



def clean_and_save_pointcloud(file_path, output_folder):
    raw_data_dict = read_vtk(file_path)
    data_dict = {}
    data_dict["points"] = raw_data_dict["points"].astype(np.float32)
    try:
        data_dict["weights"] = raw_data_dict["dnn_radius"].astype(np.float32)
    except:
        raise ValueError
    data = pv.PolyData(data_dict["points"])
    data.point_arrays["weights"] = data_dict["weights"][:,None]
    fpath = os.path.join(os.path.join(output_folder,os.path.split(file_path)[-1]))
    data.save(fpath)


def read_img(file_path, return_np=True):
    img_sitk = sitk.ReadImage(file_path)
    spacing_itk = img_sitk.GetSpacing()
    origin_itk = img_sitk.GetOrigin()
    img_np = sitk.GetArrayFromImage(img_sitk)
    if return_np:
        return img_np, np.flipud(spacing_itk),np.array([0.,0.,0])
    else:
        return img_sitk, spacing_itk, np.array([0.,0.,0])


def read_axis_reversed_img(file_path, return_np=True):
    img_sitk = sitk.ReadImage(file_path)
    spacing_itk = img_sitk.GetSpacing()
    origin_itk = img_sitk.GetOrigin()
    direction_itk = img_sitk.GetDirection()
    img_np = sitk.GetArrayFromImage(img_sitk)
    img_np = img_np[::-1]
    if return_np:
        return img_np, np.flipud(spacing_itk), np.array([0.,0.,0])
    else:
        img_sitk = sitk.GetImageFromArray(img_np)
        img_sitk.SetSpacing(spacing_itk)
        img_sitk.SetOrigin(np.array([0.,0.,0]))
        img_sitk.SetDirection(direction_itk)
        return img_sitk, spacing_itk, np.array([0.,0.,0])

def compare_dirlab_and_high_nrrd(high_pair_path,case_id = None):
    high_insp_path, high_exp_path = high_pair_path
    dir_insp_exp_shape = COPD_shape[ID_COPD[case_id]]
    high_insp_np, _, _  = read_img(high_insp_path)
    high_insp_shape = np.flipud(high_insp_np.shape)
    high_exp_np, _, _ = read_img(high_exp_path)
    high_exp_shape = np.flipud(high_exp_np.shape)
    print("{}, {} , exp_sz: {}, insp_sz: {}, copd_sz: {}, copd*4_sz: {}".format(case_id,ID_COPD[case_id], high_exp_shape[-1], high_insp_shape[-1], dir_insp_exp_shape[-1], dir_insp_exp_shape[-1]*4))


def process_high_to_dirlab(high_pair_path,case_id=None, saving_folder=None):
    high_insp_path, high_exp_path = high_pair_path
    high_insp, spacing_itk, _ = read_axis_reversed_img(high_insp_path,return_np=False)
    output_spacing = np.array(spacing_itk)
    output_spacing[-1] = output_spacing[-1]*4
    processed_insp = DataProcessing.resample_image_itk_by_spacing_and_size(high_insp, output_spacing = output_spacing, output_size=COPD_shape[ID_COPD[case_id]], output_type=None,
                                               interpolator=sitk.sitkBSpline, padding_value=0, center_padding=False)
    high_exp, spacing_itk, _ = read_axis_reversed_img(high_exp_path,return_np=False)
    processed_exp = DataProcessing.resample_image_itk_by_spacing_and_size(high_exp, output_spacing = output_spacing, output_size=COPD_shape[ID_COPD[case_id]], output_type=None,
                                               interpolator=sitk.sitkBSpline, padding_value=0, center_padding=False)
    saving_path = saving_path = os.path.join(saving_folder,case_id+"_INSP_STD_USD_COPD.nii.gz")
    sitk.WriteImage(processed_insp,saving_path)
    saving_path = saving_path = os.path.join(saving_folder, ID_COPD[case_id] +"_iBHCT.nii.gz")
    sitk.WriteImage(processed_insp,saving_path)
    saving_path = saving_path = os.path.join(saving_folder,case_id+"_EXP_STD_USD_COPD.nii.gz")
    sitk.WriteImage(processed_exp,saving_path)
    saving_path = saving_path = os.path.join(saving_folder,ID_COPD[case_id]+"_eBHCT.nii.gz")
    sitk.WriteImage(processed_exp,saving_path)





def read_landmark_index(f_path):
    """
    :param f_path: the path to the file containing the position of points.
    Points are deliminated by '\n' and X,Y,Z of each point are deliminated by '\t'.
    :return: numpy list of positions.
    """

    with open(f_path) as fp:
        content = fp.read().split('\n')

        # Read number of points from second
        count = len(content) - 1

        # Read the points
        points = np.ndarray([count, 3], dtype=np.float32)
        for i in range(count):
            if content[i] == "":
                break
            temp = content[i].split('\t')
            points[i, 0] = float(temp[0])
            points[i, 1] = float(temp[1])
            points[i, 2] = float(temp[2])
        return points


def get_img_info(img_path):
    img = sitk.ReadImage(img_path)
    origin_itk = img.GetOrigin()
    spacing_itk = img.GetSpacing()
    img_shape_itk = img.GetSize()
    return img_shape_itk, spacing_itk, origin_itk


def transfer_landmarks_from_dirlab_to_high(dirlab_index, high_shape):
    new_index= dirlab_index.copy()
    new_index[:,-1] =(high_shape[-1]- dirlab_index[:,-1]*4)+0.5
    return new_index





def process_points(point_path, img_path, case_id, output_folder, is_insp):
    index = read_landmark_index(point_path)
    img_shape_itk, spacing_itk, origin_itk = get_img_info(img_path)
    transfered_index = transfer_landmarks_from_dirlab_to_high(index, img_shape_itk)
    physical_points = transfered_index*spacing_itk+origin_itk
    data = pv.PolyData(physical_points)
    suffix = "_INSP_STD_USD_COPD.vtk" if is_insp else "_EXP_STD_USD_COPD.vtk"
    fpath = os.path.join(output_folder,case_id+suffix)
    data.save(fpath)
    suffix = "_iBHCT.vtk" if is_insp else "_eBHCT.vtk"
    fpath = os.path.join(output_folder,ID_COPD[case_id]+suffix)
    data.save(fpath)
    return fpath


def get_center(point_path,case_id, is_insp):
    points = read_vtk(point_path)["points"]
    scale, center = get_scale_and_center(points, percentile=95)
    suffix = "_INSP_STD_USD_COPD" if is_insp else "_EXP_STD_USD_COPD"
    print('"{}":{}'.format(case_id+suffix,center[0]))





save_dirlab_IMG_into_niigz = False
get_dirlab_high_shape_info = False
map_high_to_dirlab = False
project_landmarks_from_dirlab_to_high = False
pc_folder_path = "/playpen-raid1/Data/DIRLABVascular"
low_folder_path = "/playpen-raid1/Data/copd"
high_folder_path = "/playpen-raid1/Data/DIRLABCasesHighRes"
landmark_insp_key = "*_300_iBH_xyz_r1.txt"
img_insp_key = "*_iBHCT.img"
insp_key = "*_INSP_STD_*"
processed_output_path = "/playpen-raid1/Data/copd/processed"
os.makedirs(processed_output_path,exist_ok=True)
pc_insp_path_list= glob(os.path.join(pc_folder_path,insp_key))
pc_exp_path_list = [path.replace("INSP","EXP") for path in pc_insp_path_list]
for exp_path in pc_exp_path_list:
    assert os.path.isfile(exp_path),"the file {} is not exist".format(exp_path)
print("num of {} pair detected".format(len(pc_insp_path_list)))
id_list = [os.path.split(path)[-1].split("_")[0] for path in pc_insp_path_list]
landmark_insp_path_list = [os.path.join(low_folder_path,ID_COPD[_id],ID_COPD[_id],ID_COPD[_id]+"_300_iBH_xyz_r1.txt") for _id in id_list]
landmark_exp_path_list = [os.path.join(low_folder_path,ID_COPD[_id],ID_COPD[_id],ID_COPD[_id]+"_300_eBH_xyz_r1.txt") for _id in id_list]

high_img_insp_path_list = [os.path.join(high_folder_path,_id+"_INSP_STD_USD_COPD.nrrd") for _id in id_list]
high_img_exp_path_list = [os.path.join(high_folder_path,_id+"_EXP_STD_USD_COPD.nrrd") for _id in id_list]

low_processed_folder = os.path.join(processed_output_path, "dirlab")
os.makedirs(low_processed_folder, exist_ok=True)
if save_dirlab_IMG_into_niigz:
    low_img_insp_path_list = [os.path.join(low_folder_path,ID_COPD[fname],ID_COPD[fname],ID_COPD[fname]+"_iBHCT.img") for fname in id_list]
    low_img_exp_path_list = [os.path.join(low_folder_path,ID_COPD[fname],ID_COPD[fname],ID_COPD[fname]+"_eBHCT.img") for fname in id_list]
    os.makedirs(low_processed_folder)
    for low_img_insp_path, _id in zip(low_img_insp_path_list,id_list):
        save_dirlab_into_niigz(low_img_insp_path,low_processed_folder,ID_COPD[_id], is_insp=True)
    for low_img_exp_path, _id in zip(low_img_exp_path_list,id_list):
        save_dirlab_into_niigz(low_img_exp_path,low_processed_folder,ID_COPD[_id], is_insp=False)


low_img_insp_path_list = [os.path.join(low_processed_folder,_id+"_INSP_STD_USD_COPD.nii.gz") for _id in id_list]
low_img_exp_path_list = [os.path.join(low_processed_folder,_id+"_EXP_STD_USD_COPD.nii.gz") for _id in id_list]

if get_dirlab_high_shape_info:
    for i, _id in enumerate(id_list):
        high_img_insp_path, high_img_exp_path = high_img_insp_path_list[i], high_img_exp_path_list[i]
        compare_dirlab_and_high_nrrd(high_pair_path=[high_img_insp_path, high_img_exp_path],case_id = _id)

high_to_dirlab_processed_folder =  os.path.join(processed_output_path, "process_to_dirlab")
os.makedirs(high_to_dirlab_processed_folder, exist_ok=True)
if map_high_to_dirlab:
    for i, _id in enumerate(id_list):
        high_img_insp_path, high_img_exp_path = high_img_insp_path_list[i], high_img_exp_path_list[i]
        process_high_to_dirlab(high_pair_path=[high_img_insp_path, high_img_exp_path], case_id=_id, saving_folder =high_to_dirlab_processed_folder )



landmark_processed_folder =  os.path.join(processed_output_path, "landmark_processed")
os.makedirs(landmark_processed_folder, exist_ok=True)
if project_landmarks_from_dirlab_to_high:
    landmark_insp_processed_path_list = [ process_points(landmark_insp_path_list[i], high_img_insp_path_list[i], id_list[i],landmark_processed_folder, is_insp=True) for i in range(len(id_list))]
    landmark_exp_processed_path_list = [ process_points(landmark_exp_path_list[i], high_img_exp_path_list[i],id_list[i],landmark_processed_folder, is_insp=False) for i in range(len(id_list))]

# cleaned_pc_folder = os.path.join(processed_output_path, "cleaned_pointcloud")
# os.makedirs(cleaned_pc_folder, exist_ok=True)
# for i, _id in enumerate(id_list):
#     clean_and_save_pointcloud(pc_insp_path_list[i], cleaned_pc_folder)
#     clean_and_save_pointcloud(pc_exp_path_list[i], cleaned_pc_folder)

for i, _id in enumerate(id_list):
    get_center(pc_insp_path_list[i],_id, is_insp=True)
    get_center(pc_exp_path_list[i],_id,is_insp=False)

