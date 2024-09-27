# CIFAR系列数据集加载程序

数据集文件来源于：https://www.cs.utoronto.ca/~kriz/cifar.html

1. 将`cifar-10-python.tar.gz`、`cifar-100-python.tar.gz`压缩文件放置在与`cifar10.py`、`cifar100.py`同目录
2. 调用`from cifar import CIFAR10`或`CIFAR100`即可以加载数据集，获取其`numpy.ndarray`形式的数据集对象
3. 访问数据集的`train_data`、`test_data`、`labels`属性获取训练数据集、测试数据集以及全部的标签值
4. `CIFAR100`可以通过设置`coarse_labeled=True`获取粗略分类标签

# 编程示例
获取CIFAR-10数据集
```python
from cifar import CIFAR10

cifar10 = CIFAR10()
# 获取训练集和测试集
train_ds = cifar10.train_data
test_ds = cifar10.test_data
# 获取训练集和测试集的长度
len_of_train_ds = cifar10.len_of_train_ds
len_of_test_ds = cifar10.len_of_test_ds
# 获取所有标签值
labels = cifar10.labels
```
获取CIFAR-100数据集
```python
from cifar import CIFAR100

cifar100 = CIFAR100()
# 获取训练集和测试集
train_ds = cifar100.train_data
test_ds = cifar100.test_data
# 获取训练集和测试集的长度
len_of_train_ds = cifar100.len_of_train_ds
len_of_test_ds = cifar100.len_of_test_ds
# 获取所有标签值
labels = cifar100.labels
```