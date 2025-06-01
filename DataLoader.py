import numpy as np
import torch as th
import json
import torch
import datetime
import copy
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
import pickle


class MinMaxNormalization(object):
    """
        MinMax Normalization --> [-1, 1]
        x = (x - min) / (max - min).
        x = x * 2 - 1
    """

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def cal_percentage(data, percentage):
    return torch.quantile(data, percentage)
    
def clip_data(data, threshold):
    return torch.where(data > threshold, threshold, data)

def normalize_cond(cond):
    # 计算 K 维度上的最小值和最大值，保持 N, L 维度
    min_vals = cond.min(axis=0, keepdims=True).min(axis=-1, keepdims=True)  # shape (1, K, 1)
    max_vals = cond.max(axis=0, keepdims=True).max(axis=-1, keepdims=True)  # shape (1, K, 1)

    # 避免除零错误（如果 min == max，则保持原值）
    normed = (cond - min_vals) / (max_vals - min_vals + 1e-8) * 2 - 1

    return normed

def data_load_single(args, datatype):
    
    # 数据集加载 ————————————————————————————————————————————————————————————————————————————
    X, C = raw_load(datatype) # N, L
    X = X[:, :args.time_length] # N, L
    print("数据集规模：", X.shape[0])
    print("数据最大值：", X.max())

    clip_datas_train = np.percentile(X, 99)
    X = np.clip(X, a_min=None, a_max=clip_datas_train)

    args.seq_len = X.shape[-1]

    if args.prompt_state in ['load', 'test']:
        # 数据集划分
        path_idx = './indices/idx_' + datatype + '.pk'
        with open(path_idx, "rb") as f:
            train_idx, val_idx, test_idx = pickle.load(f)
        # 归一化
        path_scaler = './scalers/scaler_' + datatype + '.pk'
        with open(path_scaler, "rb") as f:
            scaler = pickle.load(f)

    elif args.prompt_state in ['train']:
        # 首先划分训练集和临时集（验证集 + 测试集）
        train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
        val_idx = test_idx
        # 归一化
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X[train_idx].reshape(-1,1))

    elif args.prompt_state in ['zero-shot']:
        # 首先划分训练集和临时集（验证集 + 测试集）
        train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.9)
        val_idx = test_idx
        # 归一化
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X[train_idx].reshape(-1,1))

    elif args.prompt_state in ['few-shot']:
        # 首先划分训练集和临时集（验证集 + 测试集）
        train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.9)
        # 归一化
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X[train_idx].reshape(-1,1))
        # 继续分隔shot集
        train_idx, test_idx = train_test_split(test_idx, test_size=1-args.fewshot_rate/0.9)
        val_idx = test_idx
    
    # 对所有子集进行标准化
    data_scaled = scaler.transform(X.reshape(-1,1)).reshape(X.shape)
    data_scaled = np.clip(data_scaled, a_min=-1, a_max=1)

    # 读取条件数据
    cond = np.array(C[:, :, :args.time_length]) # (N, K, L)
    args.feature_size = cond.shape[1]

    data = [[data_scaled[i], cond[i], X[i]] for i in range(X.shape[0])]

    # 创建子集
    dataset = MyDataset(data)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    if args.prompt_state in ['few-shot']:
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    if args.prompt_state in ['train', 'zero-shot', 'few-shot']: # 保存scaler和
        # with open('./scalers/scaler_' + datatype + '.pk', "wb") as f:
        #     pickle.dump(scaler, f)
        with open('./indices/idx_' + datatype + '.pk', "wb") as f:
            pickle.dump([train_idx, val_idx, test_idx], f)

    return  train_loader, test_loader, val_loader, scaler

def data_load_mix(args, data_list):
    data_all = []

    for data in data_list:
        data_all += data

    data_all = th.utils.data.DataLoader(data_all, batch_size=args.batch_size, shuffle=True)

    return data_all

def data_load(args):

    data_all = []
    test_data_all = []
    val_data_all = []
    my_scaler_all = {}

    for dataset_name in args.dataset.split('_'):
        data, test_data, val_data, my_scaler = data_load_single(args,dataset_name)
        data_all.append([dataset_name, data])
        test_data_all.append([dataset_name, test_data])
        val_data_all.append([dataset_name, val_data])
        my_scaler_all[dataset_name] = my_scaler

    data_all = [(name, i) for name, data in data_all for i in data]
    test_data_all = [(name, i) for name, test_data in test_data_all for i in test_data]
    val_data_all = [(name, i) for name, val_data in val_data_all for i in val_data]
    
    return data_all, test_data_all, val_data_all, my_scaler_all


def data_load_main(args):

    data, val_data, test_data, scaler = data_load(args)

    return data, test_data, val_data, scaler

def raw_load(datatype):
    
    path = '../datasets/' + datatype + '.npz'
    file = np.load(path, allow_pickle=True)
    data = file['data']

    cond = np.load("../datasets/VAE_encoded_datasets/encoded_" + datatype + ".npy", allow_pickle=True)
   
    data = torch.tensor(data, dtype=torch.float32) # (N, L)
    cond = np.array(cond, dtype=np.float32) # (N, K, L)

    data_num = 2000

    if len(data) > data_num:
        return data[:data_num], cond[:data_num]
    else:
        return data, cond