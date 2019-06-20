import random
import time
import numpy as np

def getCrossValidationIdx(length, cross_num, pad = -1):
    
    length_pad = (int(length/cross_num) + 1 ) * cross_num
    #print(length_pad)
    new_list = list(range(length)) + [pad] * (length_pad - length)
    #print(new_list)

    cross_index = np.array(random.sample(new_list, length_pad))
    #cross_index.sort()
    return cross_index.reshape((cross_num, -1))


def loadData(sents, cross_num, seed = 10, cross_validation = False, cross_idx = 0, pad = -1):
    random.seed(seed)
    if not cross_validation:
        test_index = random.sample(range(len(sents)), int(len(sents)/cross_num))
        test_index.sort()
        sentTest  = [sents[idx] for idx in l]
        sentTrain = [sents[idx] for idx in range(len(sents)) if idx not in l]
        return sentTrain, sentTest

    else:
        cross_index = getCrossValidationIdx(len(typeCCT), cross_num, pad = pad)
        test_index  = cross_index[cross_idx, ]
        sentTest    = [sents[idx] for idx in test_index]
        sentTrain   = [sents[idx] for idx in range(len(sents)) if idx not in l]
        return sentTrain, sentTest

