""" General utilities. """

import os
import shutil
from functools import partial
import torch
import random
import warnings
import requests
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import numpy as np

def init_weights(m, init_type='normal', gain=0.02):
    """ Randomly initialize a module's weights.

    Args:
        m (nn.Module): The module to initialize its weights
        init_type (str): Initialization type: 'normal', 'xavier', 'kaiming', or 'orthogonal'
        gain (float): Standard deviation of the normal distribution
    """
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'kaiming':
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            init.orthogonal_(m.weight.data, gain=gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1 or classname.find('BatchNorm3d') != -1:
        init.normal_(m.weight.data, 1.0, gain)
        init.constant_(m.bias.data, 0.0)


def set_device(gpus=None, use_cuda=True):
    """ Sets computing device. Either the CPU or any of the available GPUs.

    Args:
        gpus (list of int, optional): The GPU ids to use. If not specified, all available GPUs will be used
        use_cuda (bool, optional): If True, CUDA enabled GPUs will be used, else the CPU will be used

    Returns:
        torch.device: The selected computing device.
    """
    use_cuda = torch.cuda.is_available() if use_cuda else use_cuda
    if use_cuda:
        gpus = list(range(torch.cuda.device_count())) if not gpus else gpus
        print('=> using GPU devices: {}'.format(', '.join(map(str, gpus))))
    else:
        gpus = None
        print('=> using CPU device')
    device = torch.device('cuda:{}'.format(gpus[0])) if gpus else torch.device('cpu')

    return device, gpus


def set_seed(seed):
    """ Sets computing device. Either the CPU or any of the available GPUs.

    Args:
        gpus (list of int, optional): The GPU ids to use. If not specified, all available GPUs will be used
        use_cuda (bool, optional): If True, CUDA enabled GPUs will be used, else the CPU will be used

    Returns:
        torch.device: The selected computing device.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def save_checkpoint(exp_dir, base_name, state, is_best=False):
    """ Saves a model's checkpoint.

    Args:
        exp_dir (str): Experiment directory to save the checkpoint into.
        base_name (str): The output file name will be <base_name>_latest.pth and optionally <base_name>_best.pth
        state (dict): The model state to save.
        is_best (bool): If True, <base_name>_best.pth will be saved as well.
    """
    filename = os.path.join(exp_dir, base_name + '_latest.pth')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(exp_dir, base_name + '_best.pth'))


mag_map = {'K': 3, 'M': 6, 'B': 9}


def str2int(s):
    """ Converts a string containing a number with 'K', 'M', or 'B' to an integer. """
    if isinstance(s, (list, tuple)):
        return [str2int(o) for o in s]
    if not isinstance(s, str):
        return s
    return int(float(s[:-1]) * 10 ** mag_map[s[-1].upper()]) if s[-1].upper() in mag_map else int(s)



def sigmoid_explode(ep, static =5, k=5):
    """
    factor  increase with epoch, factor = (k + exp(ep / k))/k
    :param ep: cur epoch
    :param static: at the first #  epoch, the factor keep unchanged
    :param k: the explode factor
    :return:
    """
    static = static
    if ep < static:
        return 1.
    else:
        ep = ep - static
        factor= (k + np.exp(ep / k))/k
        return float(factor)

def sigmoid_decay(ep, static =5, k=5):
    """
    factor  decease with epoch, factor = k/(k + exp(ep / k))
    :param ep: cur epoch
    :param static: at the first #  epoch, the factor keep unchanged
    :param k: the decay factor
    :return:
    """
    static = static
    if ep < static:
        return float(1.)
    else:
        ep = ep - static
        factor =  k/(k + np.exp(ep / k))
        return float(factor)





# Adapted from: https://github.com/Sudy/coling2018/blob/master/torchtext/utils.py
def download_from_url(url, output_path):
    """ Download file from url including Google Drive.

    Args:
        url (str): File URL
        output_path (str): Output path to write the file to
    """
    def process_response(r):
        chunk_size = 16 * 1024
        total_size = int(r.headers.get('Content-length', 0))
        with open(output_path, "wb") as file:
            with tqdm(total=total_size, unit='B', unit_scale=1, desc=os.path.split(output_path)[1]) as t:
                for chunk in r.iter_content(chunk_size):
                    if chunk:
                        file.write(chunk)
                        t.update(len(chunk))

    if 'drive.google.com' not in url:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
        process_response(response)
        return

    # print('downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    process_response(response)

