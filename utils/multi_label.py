'''
Author: JustBluce 972281745@qq.com
Date: 2023-01-19 13:16:47
LastEditors: JustBluce 972281745@qq.com
LastEditTime: 2023-01-19 19:39:37
FilePath: /DialogueVersionConteol/utils/multi_label.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import torch
import numpy as np

def multilabel_categorical_crossentropy(y_pred, y_true):
#      """多标签分类的交叉熵
#      说明：y_true和y_pred的shape一致，y_true的元素非0即1，
#           1表示对应的类为目标类，0表示对应的类为非目标类。
#      警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
#           不用加激活函数，尤其是不能加sigmoid或者softmax！预测
#           阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
#           本文。
#      """
    y_pred = (1 - 2 * y_true) * y_pred # 将正例乘以-1，负例乘以1
    y_pred_neg = y_pred - y_true * 1e12 # 将正例变为负无穷，消除影响
    y_pred_pos = y_pred - (1 - y_true) * 1e12 # 将负例变为负无穷
    zeros = torch.ones_like(y_pred[..., :1], requires_grad=False)
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1) # 0阈值
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return torch.sum(neg_loss + pos_loss)


def batch_gather(input, indices):
    """
    Args:
        input: label tensor with shape [batch_size, n, L] or [batch_size, L]
        indices: predict tensor with shape [batch_size, m, l] or [batch_size, l]
    Return:
        Note that when second dimention n != m, there will be a reshape operation to gather all value along this dimention of input 
        if m == n, the return shape is [batch_size, m, l]
        if m != n, the return shape is [batch_size, n, l*m]
    """
    if indices.dtype != torch.int64:
        indices = torch.tensor(indices, dtype=torch.int64)
    results = []
    for data, index in zip(input, indices):
        if len(index) < len(data):
            index = index.reshape(-1)
            results.append(data[..., index])
        else:
            indice_dim = index.ndim
            results.append(torch.gather(data, dim=indice_dim-1, index=index))
    return torch.stack(results)


def sparse_multilabel_categorical_crossentropy(pred, label, mask_zero=False, reduction='sum'):
    """Sparse Multilabel Categorical CrossEntropy
        Reference: https://kexue.fm/archives/8888, https://github.com/bojone/bert4keras/blob/4dcda150b54ded71420c44d25ff282ed30f3ea42/bert4keras/backend.py#L272
    Args:
        label: label tensor with shape [batch_size, n, num_positive] or [Batch_size, num_positive]
            should contain the indexes of the positive rather than a ont-hot vector.
        pred: logits tensor with shape [batch_size, m, num_classes] or [batch_size, num_classes], don't use acivation.
        mask_zero: if label is used zero padding to align, please specify make_zero=True.
            when mask_zero = True, make sure the label start with 1 to num_classes, before zero padding.
    """
    zeros = torch.zeros_like(pred[..., :1])
    pred = torch.cat([pred, zeros], dim=-1)
    if mask_zero:
        infs = torch.ones_like(zeros) * np.nan
        pred = torch.cat([infs, pred], dim=-1)
    pos_2 = batch_gather(pred, label)
    pos_1 = torch.cat([pos_2, zeros], dim=-1)
    if mask_zero:
        pred = torch.cat([-infs, pred[..., 1:]], dim=-1)
        pos_2 = batch_gather(pred, label)
    pos_loss = torch.logsumexp(-pos_1, dim=-1)
    all_loss = torch.logsumexp(pred, dim=-1)
    aux_loss = torch.logsumexp(pos_2, dim=-1) - all_loss
    aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-16, 1)
    neg_loss = all_loss + torch.log(aux_loss)
    loss = pos_loss + neg_loss
    

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))