"""
这个文件用来配置项目全局参数
"""
import torch


class ParametersConfig:
    def __init__(self):
        self.train_txt = r"D:\Medical_QA_System_GPT2\local_data\medical_train.txt"
        self.valid_txt = r"D:\Medical_QA_System_GPT2\local_data\medical_valid.txt"
        self.train_pkl = r"D:\Medical_QA_System_GPT2\local_data\medical_train.pkl"
        self.valid_pkl = r"D:\Medical_QA_System_GPT2\local_data\medical_valid.pkl"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.batch_size = 4
        self.tokenizer_model = r"D:\Medical_QA_System_GPT2\models\bert-base-chinese"
        self.gpt2_pretrained_model = ''
        self.gpt2_model_config = r'D:\Medical_QA_System_GPT2\models\gpt2\config.json'
        self.save_model_path = r'D:\Medical_QA_System_GPT2\models\save_model'
        self.epochs = 4
        self.gradient_accumulation_steps = 4
        self.eps = 1.0e-09  # 为了增加数值计算的稳定性二加到分母里的项，防止在实现中分母为0
        self.warmup_steps = 1000
        self.ignore_index = -100
        self.lr = 2.6e-5
