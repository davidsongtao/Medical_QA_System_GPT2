"""
这个文件用来构建dataset,重写自定义dataset类
"""
import pickle
from torch.utils.data import Dataset
from parameters_config import *


class MyDataset(Dataset):

    # 重写MyDataset类，继承自torch.utils.data.Dataset
    def __init__(self, input_list, max_len=300):
        super().__init__()
        self.input_list = input_list
        self.max_len = max_len

    # 重写__len__方法，返回input_list的长度
    def __len__(self):
        return len(self.input_list)

    # 重写__getitem__方法，返回input_list中的索引index对应的数据
    def __getitem__(self, index):
        input_ids = self.input_list[index]
        input_ids = input_ids[:self.max_len]
        input_ids = torch.tensor(input_ids)

        return input_ids


if __name__ == '__main__':
    param = ParametersConfig()

    with open(param.train_pkl, 'rb') as f:
        train_data = pickle.load(f)
        my_dataset = MyDataset(train_data)
        print(len(my_dataset))
