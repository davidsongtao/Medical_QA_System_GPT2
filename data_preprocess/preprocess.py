"""
这个文件用来做数据预处理，将txt数据读取后切分，并进行张量表示，存入本地pkl文件
[cls]question[sep]answer[sep]
"""
import pickle

from tqdm import tqdm
from parameters_config import *
from transformers import AutoTokenizer


def preprocess(txt_path, pkl_path, tokenizer_model=ParametersConfig().tokenizer_model):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
    conversation_list = []
    conversation_length = []

    with open(txt_path, 'r', encoding='utf-8') as f:
        train_data = f.read()
        if "\r\n\r\n" in train_data:
            train_data = train_data.split("\r\n\r\n")
        else:
            train_data = train_data.split("\n\n")

        for conversation in tqdm(train_data):
            if "\r\n" in conversation:
                conversation = conversation.split("\r\n")
            else:
                conversation = conversation.split("\n")

            qa_list = [tokenizer.cls_token_id]
            for qa in conversation:
                qa_encode = tokenizer.encode(qa, add_special_tokens=False)
                qa_list += qa_encode
                qa_list.append(tokenizer.sep_token_id)

            conversation_list.append(qa_list)
            conversation_length.append(len(qa_list))

        with open(pkl_path, 'wb') as pkl:
            pickle.dump(conversation_list, pkl)

        print("数据预处理完毕，已存入本地pkl文件！")


if __name__ == '__main__':

    param = ParametersConfig()
    preprocess(param.train_txt, param.train_pkl)
    preprocess(param.valid_txt, param.valid_pkl)
