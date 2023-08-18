
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