import os
import random
import numpy as np
try:
    import pandas as pd
except Exception:
    pd = None
import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
try:
    from sklearn.preprocessing import StandardScaler
except Exception:
    class StandardScaler:
        def fit(self, data):
            data = np.asarray(data, dtype=float)
            self.mean_ = np.mean(data, axis=0)
            self.scale_ = np.std(data, axis=0)
            self.scale_[self.scale_ == 0] = 1.0
            return self

        def transform(self, data):
            data = np.asarray(data, dtype=float)
            return (data - self.mean_) / self.scale_

def _read_csv_values(path, drop_first_column=True):
    """Read numeric CSV data, falling back to NumPy when pandas is unavailable."""
    if pd is not None:
        values = pd.read_csv(path).values
    else:
        values = np.genfromtxt(path, delimiter=',', skip_header=1)
        if values.ndim == 1:
            values = values.reshape(-1, 1)

    if drop_first_column:
        values = values[:, 1:]
    return np.nan_to_num(values)


# class SWaTSegLoader(Dataset):
#     def __init__(self, data_path, win_size, step, mode="train"):
#         self.mode = mode
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()
#         data = pd.read_csv(data_path + '/train.csv', header=0, decimal=',')
#         data = data.values[:, 1:-1]

#         data = np.nan_to_num(data)
#         self.scaler.fit(data)
#         data = self.scaler.transform(data)

#         test_data = pd.read_csv(data_path + '/test.csv', delimiter=';', decimal=',')

#         y = test_data['Normal/Attack'].to_numpy()
#         labels = []
#         for i in y:
#             if i == 'Attack':
#                 labels.append(1)
#             else:
#                 labels.append(0)
#         labels = np.array(labels)

#         test_data = test_data.values[:, 1:-1]
#         test_data = np.nan_to_num(test_data)

#         self.test = self.scaler.transform(test_data)
#         self.train = data
#         self.test_labels = labels.reshape(-1, 1)

#         print("test:", self.test.shape)
#         print("train:", self.train.shape)

#     def __len__(self):
#         """
#         Number of images in the object dataset.
#         mode : "train" or "test"
#         """
#         if self.mode == "train":
#             return (self.train.shape[0] - self.win_size) // self.step + 1
#         elif self.mode == 'test':
#             return (self.test.shape[0] - self.win_size) // self.step + 1
#         else:
#             return (self.train.shape[0] - self.win_size) // self.step + 1

#     def __getitem__(self, index):
#         index = index * self.step
#         if self.mode == "train":
#             return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
#         elif self.mode == 'test':
#             return np.float32(self.test[index:index + self.win_size]), np.float32(
#                 self.test_labels[index:index + self.win_size])
#         else:
#             return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

class SWaTSegLoader(Dataset):
    """
    用于加载 SWaT 数据集的分段数据加载器。
    期望 data_path 目录下有 'SWaT_train.npy'、'SWaT_test.npy' 和 'SWaT_test_label.npy' 三个文件。
    如果不存在这些文件，会回退到 'train.npy'、'test.npy'、'test_label.npy' 或 'labels.npy'。
    """
    def __init__(self, data_path: str, win_size: int, step: int, mode: str = 'train') -> None:
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # 构造文件路径，优先使用带前缀的文件名
        train_path = os.path.join(data_path, 'SWaT_train.npy')
        test_path = os.path.join(data_path, 'SWaT_test.npy')
        label_path = os.path.join(data_path, 'SWaT_test_label.npy')

        # 若不存在前缀文件，则回退到通用文件名
        if not os.path.exists(train_path) and os.path.exists(os.path.join(data_path, 'train.npy')):
            train_path = os.path.join(data_path, 'train.npy')
        if not os.path.exists(test_path) and os.path.exists(os.path.join(data_path, 'test.npy')):
            test_path = os.path.join(data_path, 'test.npy')
        if not os.path.exists(label_path):
            if os.path.exists(os.path.join(data_path, 'test_label.npy')):
                label_path = os.path.join(data_path, 'test_label.npy')
            elif os.path.exists(os.path.join(data_path, 'labels.npy')):
                label_path = os.path.join(data_path, 'labels.npy')

        # 加载并标准化训练数据
        data = np.load(train_path)
        data = np.nan_to_num(data)  # 替换 NaN 为 0 或其他数值
        self.scaler.fit(data)
        self.train = self.scaler.transform(data)

        # 加载并标准化测试数据
        test_data = np.load(test_path)
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)

        # 加载标签，不进行缩放
        self.test_labels = np.load(label_path)

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self) -> int:
        """计算可用窗口的数量。"""
        if self.mode == 'train':
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index: int):
        """根据索引返回一个窗口的数据和对应的标签。"""
        start = index * self.step
        if self.mode == 'train':
            # 训练阶段标签占位
            return (
                np.float32(self.train[start:start + self.win_size]),
                np.float32(self.test_labels[0:self.win_size])
            )
        elif self.mode == 'test':
            return (
                np.float32(self.test[start:start + self.win_size]),
                np.float32(self.test_labels[start:start + self.win_size])
            )
        # 其他模式默认使用训练逻辑
        return (
            np.float32(self.train[start:start + self.win_size]),
            np.float32(self.test_labels[0:self.win_size])
        )

class PSMSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = _read_csv_values(os.path.join(data_path, 'train.csv'), drop_first_column=True)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = _read_csv_values(os.path.join(data_path, 'test.csv'), drop_first_column=True)

        self.test = self.scaler.transform(test_data)

        self.train = data

        self.test_labels = _read_csv_values(os.path.join(data_path, 'test_label.csv'), drop_first_column=True)

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        mode : "train" or "test"
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

class MSLSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

class SMAPSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

class SMDSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        
    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

# class WADISegLoader(Dataset):
#     def __init__(self, root_path, win_size, step, mode="train"):
#         self.mode = mode
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()

#         # train data does not have label column
#         train_data = pd.read_csv(os.path.join(root_path, 'WADI_14days_new.csv'))
#         test_data = pd.read_csv(os.path.join(root_path, 'WADI_attackdataLABLE.csv'), header=1)

#         # Remove the blank space in the columns
#         train_data.columns = [col.strip(' ') for col in train_data.columns]
#         test_data.columns = [col.strip(' ') for col in test_data.columns]

#         labels = test_data["Attack LABLE (1:No Attack, -1:Attack)"].apply(lambda x: 0 if x == 1 else 1).values
#         test_data = test_data.drop(['Attack LABLE (1:No Attack, -1:Attack)'], axis=1)

#         # Removing columns that only contain time and nans (data missing from the csv file)
#         train_nan_columns = {col for col in train_data.columns if train_data[col].isna().all()}
#         test_nan_columns = {col for col in test_data.columns if test_data[col].isna().all()}
#         common_nan_columns = list(train_nan_columns.intersection(test_nan_columns))

#         train_data = train_data.drop(['Row', 'Date', 'Time'] + common_nan_columns, axis=1)
#         test_data = test_data.drop(['Row', 'Date', 'Time'] + common_nan_columns, axis=1)

#         # fill the missing values
#         train_data = train_data.interpolate().bfill()
#         test_data = test_data.interpolate().bfill()

#         train_data = train_data.values
#         test_data = test_data.values

#         self.scaler.fit(train_data)
#         train_data = self.scaler.transform(train_data)
#         test_data = self.scaler.transform(test_data)

#         data_len = len(train_data)
#         self.train = train_data[:int(data_len * 0.8)]
#         self.val = train_data[int(data_len * 0.8):]
#         self.test = test_data
#         self.test_labels = labels
#         print("train:", self.train.shape)
#         print("test:", self.test.shape)

#     def __len__(self):
#         """
#         Number of images in the object dataset.
#         """
#         if self.mode == "train":
#             return (self.train.shape[0] - self.win_size) // self.step + 1
#         elif self.mode == 'val':
#             return (self.val.shape[0] - self.win_size) // self.step + 1
#         elif self.mode == 'test':
#             return (self.test.shape[0] - self.win_size) // self.step + 1
#         else:
#             return (self.test.shape[0] - self.win_size) // self.win_size + 1

#     def __getitem__(self, index):
#         index = index * self.step
#         if self.mode == "train":
#             return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
#         elif self.mode == 'val':
#             return np.float32(self.val[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
#         elif self.mode == 'test':
#             return np.float32(self.test[index:index + self.win_size]), np.float32(
#                 self.test_labels[index:index + self.win_size])
#         else:
#             return np.float32(self.test[
#                               index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
#                 self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class WADISegLoader(Dataset):
    """
    用于加载 WADI 数据集的分段数据加载器。会优先使用 WADI_train.npy、
    WADI_test.npy 和 WADI_test_label.npy，如果不存在则退回到通用文件名。
    加载训练集并拟合 StandardScaler，然后对训练集和测试集应用同一标准化。
    """

    def __init__(self, data_path: str, win_size: int, step: int, mode: str = 'train') -> None:
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # 默认文件名
        train_path = os.path.join(data_path, 'WADI_train.npy')
        test_path = os.path.join(data_path, 'WADI_test.npy')
        label_path = os.path.join(data_path, 'WADI_test_label.npy')

        # 若不存在则回退到通用文件名
        if not os.path.exists(train_path) and os.path.exists(os.path.join(data_path, 'train.npy')):
            train_path = os.path.join(data_path, 'train.npy')
        if not os.path.exists(test_path) and os.path.exists(os.path.join(data_path, 'test.npy')):
            test_path = os.path.join(data_path, 'test.npy')
        if not os.path.exists(label_path):
            if os.path.exists(os.path.join(data_path, 'test_label.npy')):
                label_path = os.path.join(data_path, 'test_label.npy')
            elif os.path.exists(os.path.join(data_path, 'labels.npy')):
                label_path = os.path.join(data_path, 'labels.npy')

        # 加载并标准化训练数据
        data = np.load(train_path)
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        self.train = self.scaler.transform(data)

        # 加载并标准化测试数据
        test_data = np.load(test_path)
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)

        # 加载标签，若为 1 维则转成列向量
        labels = np.load(label_path)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        self.test_labels = labels

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self) -> int:
        """计算窗口数量"""
        if self.mode == 'train':
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index: int):
        """返回窗口数据及标签"""
        start = index * self.step
        if self.mode == 'train':
            return (
                np.float32(self.train[start:start + self.win_size]),
                np.float32(self.test_labels[0:self.win_size])
            )
        elif self.mode == 'test':
            return (
                np.float32(self.test[start:start + self.win_size]),
                np.float32(self.test_labels[start:start + self.win_size])
            )
        return (
            np.float32(self.train[start:start + self.win_size]),
            np.float32(self.test_labels[0:self.win_size])
        )

class NIPS_TS_WaterSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Water_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_Water_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_Water_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class NIPS_TS_SwanSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Swan_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_Swan_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_Swan_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class NIPS_TS_CCardSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_creditcard_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_creditcard_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_creditcard_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD', val_ratio=0.2):
    """
    model : 'train' or 'test'
    """

    if dataset == 'SMD':
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif dataset == 'MSL':
        dataset = MSLSegLoader(data_path, win_size, step, mode)
    elif dataset == 'SMAP':
        dataset = SMAPSegLoader(data_path, win_size, step, mode)
    elif dataset == 'PSM':
        dataset = PSMSegLoader(data_path, win_size, step, mode)
    elif dataset == 'SWaT':
        dataset = SWaTSegLoader(data_path, win_size, step, mode)
    elif dataset == 'WADI':
        dataset = WADISegLoader(data_path, win_size, step, mode)
    elif dataset == 'NIPS_TS_Water':
        dataset = NIPS_TS_WaterSegLoader(data_path, win_size, step, mode)
    elif dataset == 'NIPS_TS_Swan':
        dataset = NIPS_TS_SwanSegLoader(data_path, win_size, step, mode)
    elif dataset == 'NIPS_TS_Creditcard':
        dataset = NIPS_TS_CCardSegLoader(data_path, win_size, step, mode)

    shuffle = False

    if mode == 'train':
        shuffle = True

        dataset_len = int(len(dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))

        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)

        indices = torch.arange(dataset_len)
        
        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(dataset, val_sub_indices)
        
        train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=shuffle)

        # k_use_len = int(train_use_len*0.1)
        # k_sub_indices = indices[:k_use_len]
        # k_subset = Subset(dataset, k_sub_indices)
        # k_loader = DataLoader(dataset=k_subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        #
        # return train_loader, val_loader, k_loader

        return train_loader, val_loader

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
