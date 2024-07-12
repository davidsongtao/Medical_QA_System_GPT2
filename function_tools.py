"""
功能性方法如：计算损失函数损失值，计算模型输出准确率等
"""


def calculate_acc(logits, labels, ignore_index):
    """
    计算模型准确率
    :param logits: 传入的模型预测值
    :param labels: 真实的标签
    :param ignore_index: 忽略的索引值
    :return: batch_acc: 模型准确率
    """
    # 1. 模型输出值和真实标签值进行切片，形状变换，以满足准确率计算
    logit = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)
    # 2. 从变换形状的预测值中选取概率最大的结果，即真实预测结果
    _, logit = logit.max(dim=-1)
    # 3. 真实标签中去掉填充的无意义的内容
    non_pad_mask = labels.ne(ignore_index)
    # 4. 计算模型输出结果中预测正确的结果数
    batch_num_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    # 5. 计算模型输出结果中预测的总数
    batch_num_total = non_pad_mask.sum().item()
    # 6. 计算模型准确率
    batch_acc = batch_num_correct / batch_num_total
    return batch_acc


