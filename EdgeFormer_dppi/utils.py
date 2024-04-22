import numpy as np
import scipy.sparse as sp
import torch
import random
from torch.utils import data
from torch.utils.data import DataLoader, SubsetRandomSampler

def load_train_data(path, dataset):
    #print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features1_dppi = sp.csr_matrix(idx_features_labels[:, 1:21], dtype=np.float32)
    features2_dpip = sp.csr_matrix(idx_features_labels[:, -21:-1], dtype=np.float32)
    labels = idx_features_labels[:, -1]

    features1_dppi = torch.FloatTensor(np.array(features1_dppi.todense()))
    features2_dpip = torch.FloatTensor(np.array(features2_dpip.todense()))
    labels = torch.LongTensor(labels.astype(int))

    # 找出label为1的下标，并随机取取对应大小的label=0的下标
    idx_train = []
    for index, value in enumerate(labels):
        if value == 1:
            idx_train.append(index)

    len1 = len(idx_train)
    for index, value in enumerate(labels):
        temp = random.randint(0, len(labels) - 1)
        if labels[temp] == 0:
            if (len(idx_train) < (int)(2 * len1)):
                idx_train.append(temp)
    return features1_dppi, features2_dpip, labels, idx_train

def load_test_data(path, dataset):
    #print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features1_dppi = sp.csr_matrix(idx_features_labels[:, 1:21], dtype=np.float32)
    features2_dpip = sp.csr_matrix(idx_features_labels[:, -21:-1], dtype=np.float32)
    labels = idx_features_labels[:, -1]

    features1_dppi = torch.FloatTensor(np.array(features1_dppi.todense()))
    features2_dpip = torch.FloatTensor(np.array(features2_dpip.todense()))
    labels = torch.LongTensor(labels.astype(int))
    return features1_dppi, features2_dpip, labels

def load_test_data_no_labels(path, dataset):
    #print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features1_dppi = sp.csr_matrix(idx_features_labels[:, 1:21], dtype=np.float32)
    # 注意修改这里，无label
    features2_dpip = sp.csr_matrix(idx_features_labels[:, -21:-1], dtype=np.float32)

    features1_dppi = torch.FloatTensor(np.array(features1_dppi.todense()))
    features2_dpip = torch.FloatTensor(np.array(features2_dpip.todense()))
    return features1_dppi, features2_dpip

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def back_loader(str, batch_size):
    path = "./datasets/" + str + "_datasets/"
    str_split = path.split("/")
    dataset = str_split[len(str_split) - 2]

    if str == 'train':
        features1_dppi, features2_dpip, labels, idx_train = load_train_data(path=path, dataset=dataset)

        dataset = data.TensorDataset(features1_dppi, features2_dpip, labels)
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,  # 每批提取的数量
            shuffle=False,  # 要不要打乱数据（打乱比较好）
            num_workers=0,  # 多少线程来读取数据
            drop_last=True,
            sampler=SubsetRandomSampler(idx_train)
        )
    else:
        features1_dppi, features2_dpip, labels = load_test_data(path=path, dataset=dataset)
        dataset = data.TensorDataset(features1_dppi, features2_dpip, labels)
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,  # 每批提取的数量
            shuffle=False,  # 要不要打乱数据（打乱比较好）
            num_workers=0,  # 多少线程来读取数据
            drop_last=False
        )
    return loader