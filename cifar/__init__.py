import pickle
import numpy as np


def extract_cifar_tar(tar_file_path, filter_rules):
    import tarfile
    with tarfile.open(tar_file_path, 'r:gz') as tar:
        # 获取文件名
        file_names = filter(filter_rules, tar.getnames())
        # 将文件进行解压
        files = [tar.extractfile(tar.getmember(name)) for name in file_names]
        # 将文件反序列化以获取可读形式数据
        return [pickle.load(file, encoding='bytes') for file in files]


def convert_to_RGB(raw_data):
    def impl(img_array):
        r_array, g_array, b_array = img_array[:1024], img_array[1024:2048], img_array[2048:]
        return np.stack([
            r_array.reshape(32, 32).swapaxes(0, 1),
            g_array.reshape(32, 32).swapaxes(0, 1),
            b_array.reshape(32, 32).swapaxes(0, 1)
        ]).swapaxes(0, 2)

    return np.stack([impl(rd) for rd in raw_data])


from .cifar100 import CIFAR100
from .cifar10 import CIFAR10
