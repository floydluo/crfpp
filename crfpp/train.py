import os
import time 
import optparse 
import pickle
from pprint import pprint

import pandas as pd

from crfpp.tagger import tagger
from crfpp.loaddata import loadData
from crfpp.crftools import get_dfsent_strfeats, crf_learn, crf_test
from crfpp.evals import get_sent_annoSET, match_anno_pred_result, calculate_F1_Score, logError

def crfpp_learn(sents, Channel_Settings, model_path, ):
    # create para
    para = {}
    feats_data_path = '_tmp/_crfpp_train_feats.txt'
    template_path   = '_tmp/_template'
    DFtrain = pd.DataFrame()
    for sent in sents:
        df = get_dfsent_strfeats(sent, Channel_Settings) #
        df.loc[len(df)] = np.NaN     ## Trick Here
        DFtrain = DFtrain.append(df) ## Trick Here
    DFtrain = DFtrain.reset_index(drop=True)
    DFtrain.to_csv(feats_data_path, sep = '\t', encoding = 'utf=8', header = False, index = False )

    generate_template(gram_1 = DFtrain.shape[1] - 1, path = template_path) # this needs attention
    crf_learn(feats_data_path, model_path, template_path  = template_path)
    return para


def crfpp_test(sents, Channel_Settings, model_path, label_list):
    '''
        sents: a list of sents
        This function could be updated in the future for a better performance.
        But currently, just leave it now.
    '''
    pred_entities = []
    anno_entities = []
    log_list     = []
    # here sents are a batch of sents, not necessary to be all the sents
    # this could be chanage, but it will cause some time, so just ignore it.
    for sent in sents:
        pred_SET = tagger(sent, model_path, Channel_Settings)
        pred_entities.append(pred_SET)
        anno_SET = get_sent_annoSET(sent)
        anno_entities.append(anno_SET)
        error = logError(sent, pred_SET, anno_SET)
        log_list.append(error)
    
    # return anno_entities, pred_entities
    Result = match_anno_pred_result(anno_entities, pred_entities, label_list = label_list)
    # return Result
    R = calculate_F1_Score(Result, label_list)

    LogError = pd.concat(log_list).reset_index(drop = True)

    return R, LogError


def trainModel(model_path, sentTrain, sentTest, Channel_Settings, label_list):
    '''
        sentTrain, sentTest: are featurized already.
    '''
    para   = crfpp_learn(sentTrain, Channel_Settings, model_path)
    R, Err = crfpp_test (sentTest,  Channel_Settings, model_path, label_list)
    # generate the error log path
    return R

def train(sents, Channel_Settings, label_list, cross_num, cross_validation = None, seed = 10):
    '''
        sent is featurized
    '''
    if not cross_validation:
        sentTrain, sentTest = loadData(sents, cross_num, seed = 10, cross_validation = cross_validation, cross_idx = 0)
        Performance = trainModel(model_path, sentTrain, sentTest, Channel_Settings, label_list)
        print('\nThe Final Performance is:\n')
        return Performance
    else:
        L = []
        for cross_idx in range(cross_num):
            R = trainModel(model_path, sentTrain, sentTest, Channel_Settings, label_list)
            L.append(R)
        Performance = sum(L)/cross_num
        print('\nThe Final Average Performance for', cross_num, 'Cross Validation is:\n')
        print(Performance)
        return Performance