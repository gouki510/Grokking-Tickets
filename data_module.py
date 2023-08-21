import random
import numpy as np


def gen_train_test(frac_train, num, seed=0, is_symmetric_input=False):
    # Generate train and test split
    if is_symmetric_input:
      pairs = [(i, j) for i in range(num) for j in range(num) if i <= j]
    else:
      pairs = [(i, j) for i in range(num) for j in range(num)]
    random.seed(seed)
    random.shuffle(pairs)
    div = int(frac_train*len(pairs))
    return pairs[:div], pairs[div:]

def train_test_split(p,train,test):
    is_train = []
    is_test = []
    for x in range(p):
        for y in range(p):
            if (x, y) in train:
                is_train.append(True)
                is_test.append(False)
            else:
                is_train.append(False)
                is_test.append(True)
    is_train = np.array(is_train)
    is_test = np.array(is_test)
    return is_train, is_test