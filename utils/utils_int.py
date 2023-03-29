'''
Author: JustBluce 972281745@qq.com
Date: 2023-01-12 14:36:23
LastEditors: JustBluce 972281745@qq.com
LastEditTime: 2023-01-12 15:25:16
FilePath: /DialogueVersionConteol/utils/random_int.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import random
import copy

def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list


def minimum_int_list(output, n):
    t = copy.deepcopy(output)
    # 求m个最大的数值及其索引
    min_number = []
    min_index = []
    for _ in range(n):
        number = min(t)
        index = t.index(number)
        t[index] = 1000000000000000
        min_number.append(number)
        min_index.append(index)
    t = []
    return min_index