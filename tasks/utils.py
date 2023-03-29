'''
Author: JustBluce 972281745@qq.com
Date: 2022-11-24 08:58:24
LastEditors: JustBluce 972281745@qq.com
LastEditTime: 2023-02-16 11:29:18
FilePath: /ATIS/tasks/utils.py
Description: 关于一些数据集和模型的设置。
'''

VQA_TASKS = ["flickr"]

DATASETS = VQA_TASKS


USE_FAST = {
    'bert': True,
    'roberta': True,
    'deberta': True,
    'gpt2': True,
    'deberta-v2': False,
    'blip2': True
}