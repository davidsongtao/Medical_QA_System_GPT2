"""
这个文件用来构建自定义dataloader，为模型输入做准备
"""
from torch.nn.utils import rnn
from torch.utils.data import DataLoader
from data_preprocess.dataset import *


def load_dataset(train_pkl, valid_pkl):
    with open(train_pkl, 'rb') as f:
        train_data = pickle.load(f)
        train_dataset = MyDataset(train_data, max_len=300)

    with open(valid_pkl, 'rb') as f:
        valid_data = pickle.load(f)
        valid_dataset = MyDataset(valid_data, max_len=300)

    return train_dataset, valid_dataset


def collate_fn(batch):
    input_ids = rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn.pad_sequence(batch, batch_first=True, padding_value=-100)

    return input_ids, labels


def get_dataloader(train_pkl, valid_pkl):

    param = ParametersConfig()
    train_dataset, valid_dataset = load_dataset(train_pkl, valid_pkl)

    train_dataloader = DataLoader(train_dataset, batch_size=param.batch_size, shuffle=True, drop_last=True,
                                  collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=param.batch_size, shuffle=False, drop_last=True,
                                  collate_fn=collate_fn)

    return train_dataloader, valid_dataloader
