from cifar import extract_cifar_tar, convert_to_RGB
import os
import numpy as np

class CIFAR10:
    """CIFAR10数据集对象"""

    def __init__(self):
        """CIFAR10数据集对象
        加载数据集，并对原始数据进行整理
        """
        # 获取当前脚本的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建数据集文件的绝对路径
        tar_file_path = os.path.join(current_dir, 'cifar-10-python.tar.gz')

        assert os.path.exists(tar_file_path), 'CIFAR10数据集文件丢失！'
        # 解包文件
        (train_batch4, test_batch, train_batch3, label_names, train_batch2,
         train_batch5, train_batch1) = extract_cifar_tar(
            tar_file_path,
            lambda s: 'data' in s or 'meta' in s or 'test' in s
        )
        self.__train_batch = {
            'features': np.vstack([
                convert_to_RGB(batch[b'data'])
                for batch in [train_batch1, train_batch2, train_batch3, train_batch4, train_batch5]
            ]),
            'labels': np.hstack([
                batch[b'labels']
                for batch in [train_batch1, train_batch2, train_batch3, train_batch4, train_batch5]
            ])
        }
        self.__test_batch = {
            'features': test_batch[b'data'], 'labels': test_batch[b'labels']
        }
        self.__labels = label_names[b'label_names']

    @property
    def train_data(self):
        return self.__train_batch

    @property
    def test_data(self):
        return self.__test_batch

    @property
    def labels(self):
        return self.__labels
