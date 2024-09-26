from cifar import extract_cifar_tar, convert_to_RGB
import os


class CIFAR100:
    """CIFAR100数据集对象"""

    def __init__(self, coarse_labeled=False):
        """CIFAR100数据集对象
        加载数据集，并对原始数据进行整理
        :param coarse_labeled: 是否需要粗略分类标签
        """
        # 获取当前脚本的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建数据集文件的绝对路径
        tar_file_path = os.path.join(current_dir, 'cifar-100-python.tar.gz')

        assert os.path.exists(tar_file_path), 'CIFAR100数据集文件丢失！'
        # 解包文件
        train_batch, test_batch, label_names = extract_cifar_tar(
            tar_file_path,
            lambda s: 'meta' in s or 'train' in s or 'test' in s
        )
        # 将矩阵数据整理好
        train_batch[b'data'] = convert_to_RGB(train_batch[b'data'])
        test_batch[b'data'] = convert_to_RGB(test_batch[b'data'])
        # 去除多余部分数据
        if coarse_labeled:
            self.__train_batch = {
                'features': train_batch[b'data'], 'labels': train_batch[b'coarse_labels']
            }
            self.__test_batch = {
                'features': test_batch[b'data'], 'labels': test_batch[b'coarse_labels']
            }
            self.__labels = label_names[b'coarse_label_names']
        else:
            self.__train_batch = {
                'features': train_batch[b'data'], 'labels': train_batch[b'fine_labels']
            }
            self.__test_batch = {
                'features': test_batch[b'data'], 'labels': test_batch[b'fine_labels']
            }
            self.__labels = label_names[b'fine_label_names']

    @property
    def train_data(self):
        return self.__train_batch

    @property
    def test_data(self):
        return self.__test_batch

    @property
    def labels(self):
        return self.__labels
