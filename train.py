"""
模型训练脚本
"""
import os
import transformers
from function_tools import *
from parameters_config import *
from transformers import BertTokenizerFast, GPT2LMHeadModel, GPT2Config
from data_preprocess.dataloader import *


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
    epoch_loss = 0
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
        epoch_loss += loss
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
            print(
                f"模型当前轮次：{batch_index + 1}. 模型当前损失值：{loss * param.gradient_accumulation_steps}. 模型当前准确率：{batch_acc}")
        # 10. 清空内存
        del input_ids, outputs

    # 保存模型
    # if epoch % 1 == 0 or epoch == param.epochs:
    #     model_path = os.path.join(param.save_model_path, f"model_epoch{epoch + 1}")
    #     if not os.path.exists(model_path):
    #         os.mkdir(model_path)
    #     model.save_pretrained(model_path)
    #     print(f"模型第{epoch + 1}轮。保存成功！")


def valid_epoch(model, valid_dataloader, param, epoch):
    # 1. 指定模型进入验证模式
    model.eval()
    total_loss = 0
    # 2. 从验证集解析数据，丢入模型
    with torch.no_grad():
        for batch_index, (input_ids, labels) in enumerate(valid_dataloader):
            # 3. 数据拉入GPU
            input_ids = input_ids.to(param.device)
            labels = labels.to(param.device)
            # 4. 拉入模型获取输出结果
            outputs = model.forward(input_ids, labels=labels)
            # 5. 结果解析出loss
            loss = outputs.loss.mean()
            total_loss += loss
        # 6. 计算epoch的平均损失值
        epoch_mean_loss = total_loss / len(valid_dataloader)
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
    # 4. 开始模型训练
    for epoch in range(param.epochs):
        epoch_best_loss = 10000
        train_epoch(model, train_dataloader, optimizer, scheduler, param, epoch)
        # 5. 模型在验证集上验证
        epoch_mean_loss = valid_epoch(model, valid_dataloader, param, epoch)
        if epoch_mean_loss < epoch_best_loss:
            epoch_best_loss = epoch_mean_loss
            # 保存模型
            best_model_path = os.path.join(param.save_model_path, f"best_model_epoch_{epoch}")
            if not os.path.exists(best_model_path):
                os.mkdir(best_model_path)
            model.save_pretrained(best_model_path)
            print("最佳模型保存成功！")


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
