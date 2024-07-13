"""
模型训练脚本
"""
import os
from datetime import datetime

import transformers
from function_tools import *
from parameters_config import *
from transformers import BertTokenizerFast, GPT2LMHeadModel, GPT2Config
from data_preprocess.dataloader import *
from data_preprocess.dataset import *


def train_epoch(model, train_dataloader, optimizer, scheduler, param, epoch):
    """
    模型训练函数
    :param model: GPT2模型
    :param train_dataloader: 训练集dataloader
    :param optimizer: 优化器，用来迭代模型参数
    :param scheduler: 学习率预热
    :param param: 超参
    """
    # 1. 设置模型进入训练模式
    model.train()
    epoch_total_loss = 0
    # 2. train_dataloader中解析出input_ids和labels
    for batch_index, (input_ids, labels) in enumerate(train_dataloader):
        # 3. 数据拉入显存
        input_ids = input_ids.to(param.device)
        labels = labels.to(param.device)
        # 4. 前向传播，获取模型输出
        outputs = model.forward(input_ids, labels=labels)
        # 5. 模型输出结果中解析出预测值，损失值
        logits = outputs.logits
        loss = outputs.loss.mean()
        # 计算这个轮次的总损失值
        epoch_total_loss += loss
        ignore_index = param.ignore_index
        # 6. 通过模型输出预测值和损失值计算模型准确率
        batch_acc = calculate_acc(logits, labels, ignore_index)
        # 7. 当梯度累加大于1时，计算出真实的损失值
        if param.gradient_accumulation_steps > 1:
            loss = loss / param.gradient_accumulation_steps
        # 8. 反向传播，模型参数更新
        loss.backward()
        if (batch_index + 1) % param.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        # 9. 打印日志
        if (batch_index + 1) % param.loss_step == 0:
            loss_print = loss * param.gradient_accumulation_steps
            loss_print = loss_print.item()
            acc_print = round(batch_acc, 4)*100
            with open(param.train_log_path, "a") as train_log:
                print(
                    f"当前Batch：{batch_index + 1}/{len(train_dataloader)} | 当前epoch:{epoch}/{param.epochs} | 当前损失值：{round(loss_print, 4)} | 当前准确率：{acc_print:.2f}%", file=train_log)
            print(
                f"当前Batch：{batch_index + 1}/{len(train_dataloader)} | 当前epoch:{epoch}/{param.epochs} | 当前损失值：{round(loss_print, 4)} | 当前准确率：{acc_print:.2f}%")
        # 10. 清空内存
        del input_ids, outputs
    # 计算这个epoch的平均损失值
    epoch_mean_loss = epoch_total_loss / len(train_dataloader)
    return epoch_mean_loss


def valid_epoch(model, valid_dataloader, param, epoch):
    """
    用训练的模型在验证集上验证，计算损失值
    :param model: 训练集上训练过的模型
    :param valid_dataloader: 验证数据集
    :param param: 超参
    :param epoch: 当前轮次
    :return: epoch_mean_loss:本轮次验证集平均损失
    """
    # 1. 指定模型进入验证模式
    model.eval()
    # 初始化epoch_total_loss
    epoch_total_loss = 0
    # 2. 解析数据集进行验证
    with torch.no_grad():
        for batch_index, (input_ids, labels) in enumerate(valid_dataloader):
            # 3. 数据拉入GPU
            input_ids = input_ids.to(param.device)
            labels = labels.to(param.device)
            # 4. 数据拉入模型前向传播
            outputs = model.forward(input_ids, labels=labels)
            # 5. 输出结果解析出损失值
            loss = outputs.loss.mean()
            epoch_total_loss += loss
        # 6. 计算验证集这个epoch的平均loss
        epoch_mean_loss = epoch_total_loss / len(valid_dataloader)
        return epoch_mean_loss


def train(model, train_dataloader, valid_dataloader, param):
    """
    模型训练前置准备，构建优化器，学习率预热，等
    :param model: GPT2模型
    :param train_dataloader: 训练集dataloader
    :param valid_dataloader: 验证集dataloader
    :param param: 超参
    """
    # 1. 计算损失迭代总步数
    t_total = len(train_dataloader) // param.gradient_accumulation_steps * param.epochs
    # 2. 构建优化器
    optimizer = transformers.AdamW(model.parameters(), lr=param.lr, eps=param.eps, no_deprecation_warning=True)
    # 3. 构建学习率预热
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=param.warmup_steps,
                                                             num_training_steps=t_total)
    # 初始化两个空列表，记录训练集和验证集各轮次的平均损失值
    train_losses, valid_losses = [], []
    # 初始化一个最佳损失值，后续迭代更新
    epoch_best_loss = 10000
    # 4. 开始模型训练
    with open(param.train_log_path, "a") as train_log:
        print(f"模型训练开始，起始时间：{datetime.now()}", file=train_log)
    print(f"模型训练开始，起始时间：{datetime.now()}")
    for epoch in range(param.epochs):
        # 5. 数据拉入训练集训练
        train_mean_loss = train_epoch(model, train_dataloader, optimizer, scheduler, param, epoch)
        train_losses.append(train_mean_loss)
        # 6. 模型在验证集上验证
        valid_mean_loss = valid_epoch(model, valid_dataloader, param, epoch)
        valid_losses.append(valid_mean_loss)
        # 7. 保存最佳模型
        if valid_mean_loss < epoch_best_loss:
            epoch_best_loss = valid_mean_loss
            best_model_path = os.path.join(param.save_model_path, f"best_model_{epoch}")
            if not os.path.exists(best_model_path):
                os.mkdir(best_model_path)
            model.save_pretrained(best_model_path)
            with open(param.train_log_path, "a") as train_log:
                print("最佳模型保存成功！", file=train_log)
            print("最佳模型保存成功！")

    # 8. 训练结束，打印以下日志
    with open(param.train_log_path, "a") as train_log:
        print(f"所有训练结束。结束时间：{datetime.now()}", file=train_log)
        print(f"训练集损失值：{train_losses}", file=train_log)
        print(f"验证集损失值：{valid_losses}", file=train_log)
        print(f"最佳损失值：{epoch_best_loss}", file=train_log)
    print(f"所有训练结束。结束时间：{datetime.now()}")
    print(f"训练集损失值：{train_losses}")
    print(f"验证集损失值：{valid_losses}")
    print(f"最佳损失值：{epoch_best_loss}")


def main():
    """
    模型训练主函数，进行模型训练
    :return:
    """
    # 1. 实例化超参
    param = ParametersConfig()
    # 2. 设置使用的显存设备
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # 3. 实例化分词器
    tokenizer = BertTokenizerFast.from_pretrained(param.tokenizer_model)
    # 4. 创建模型储存路径
    if not os.path.exists(param.save_model_path):
        os.mkdir(param.save_model_path)
    # 5. 构建GPT2模型
    if param.gpt2_pretrained_model:
        model = GPT2LMHeadModel.from_pretrained(param.gpt2_pretrained_model)
    else:
        config = GPT2Config.from_json_file(param.gpt2_model_config)
        model = GPT2LMHeadModel(config=config)
    # 6. 模型拉入显存
    model.to(param.device)
    # 7. 确认分词器词典大小和模型词典大小一样
    assert tokenizer.vocab_size == model.config.vocab_size
    # 8. 构建训练数据集和验证数据集，准备开始训练
    train_dataloader, valid_dataloader = get_dataloader(param.train_pkl, param.valid_pkl)
    # 9. 开始模型训练
    train(model, train_dataloader, valid_dataloader, param)


if __name__ == '__main__':
    main()
