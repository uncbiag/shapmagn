from glob import glob
import os
import numpy as np
import pyvista as pv
import SimpleITK as sitk


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

def process_IMG(file_path, shape, fname):
    dtype = np.dtype("<i2")
    fid = open(file_path, 'rb')
    data = np.fromfile(fid, dtype)
    image = data.reshape(shape)

    return image


def readPoint(f_path):
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


def get_origin(img_path):
    img = sitk.ReadImage(img_path)
    origin = img.GetOrigin()
    return origin

def process_points(point_path, img_path, name, output_folder, is_insp):
    points = readPoint(point_path)
    copd = ID_COPD[name]
    ori_spacing =COPD_spacing[copd]
    origin = get_origin(img_path)
    physical_points = (points-1)*ori_spacing + origin
    data = pv.PolyData(physical_points)
    suffix = "_insp" if is_insp else "_exp"
    fpath = os.path.join(output_folder,name+ suffix+".vtk")
    data.save(fpath)
    return fpath


data_folder_path = "/playpen-raid1/Data/DIRLABVascular"
landmark_folder_path = "/playpen-raid1/Data/copd"
img_folder_path = "/playpen-raid1/Data/DIRLABCasesHighRes"
landmark_insp_key = "*_300_iBH_xyz_r1.txt"
insp_key = "*_INSP_STD_*"
processed_output_path = "/playpen-raid1/Data/copd/processed"
os.makedirs(processed_output_path,exist_ok=True)
insp_path_list= glob(os.path.join(data_folder_path,insp_key))
exp_path_list = [path.replace("INSP","EXP") for path in insp_path_list]
for exp_path in exp_path_list:
    assert os.path.isfile(exp_path),"the file {} is not exist".format(exp_path)
print("num of {} pair detected".format(len(insp_path_list)))
file_name_list = [os.path.split(path)[-1].split("_")[0] for path in insp_path_list]
landmark_insp_path_list = [os.path.join(landmark_folder_path,ID_COPD[fname],ID_COPD[fname],ID_COPD[fname]+"_300_iBH_xyz_r1.txt") for fname in file_name_list]
landmark_exp_path_list = [os.path.join(landmark_folder_path,ID_COPD[fname],ID_COPD[fname],ID_COPD[fname]+"_300_eBH_xyz_r1.txt") for fname in file_name_list]
img_insp_path_list = [os.path.join(img_folder_path,fname+"_INSP_STD_USD_COPD.nrrd") for fname in file_name_list]
img_exp_path_list = [os.path.join(img_folder_path,fname+"_EXP_STD_USD_COPD.nrrd") for fname in file_name_list]
landmark_insp_processed_path_list = [ process_points(landmark_insp_path_list[i], img_insp_path_list[i], file_name_list[i],processed_output_path, is_insp=True) for i in range(len(file_name_list))]
landmark_exp_processed_path_list = [ process_points(landmark_exp_path_list[i], img_exp_path_list[i],file_name_list[i],processed_output_path, is_insp=False) for i in range(len(file_name_list))]




insp_name_list = [name+"_insp" for name in file_name_list]
exp_name_list = [name+"_exp" for name in file_name_list]
insp_list = [{"name":insp_name, "data_path":insp_path,"landmark_path":lnk_path}
             for insp_name, insp_path, lnk_path in zip(insp_name_list, insp_path_list,landmark_insp_processed_path_list)]
exp_list = [{"name":exp_name, "data_path":exp_path,"landmark_path":lnk_path}
             for exp_name, exp_path , lnk_path in zip(exp_name_list, exp_path_list,landmark_exp_processed_path_list)]

pair_name_list = [insp_name + "_"+ exp_name for insp_name, exp_name in zip(insp_name_list,exp_name_list)]
output_dict = {pair_name: {"source": source,"target":target} for pair_name, source, target in zip(pair_name_list,exp_list,insp_list)}

