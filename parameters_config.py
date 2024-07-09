"""
这个文件用来配置项目全局参数
"""
import torch


class ParametersConfig:
    def __init__(self):
        self.train_txt = r"../local_data/medical_train.txt"
        self.valid_txt = r"../local_data/medical_valid.txt"
        self.train_pkl = r"../local_data/medical_train.pkl"
        self.valid_pkl = r"../local_data/medical_valid.pkl"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.batch_size = 4
        self.tokenizer_model = r"../models/bert-base-chinese"
