import os
import time 
import optparse 
import pickle
from pprint import pprint
from datetime import datetime
import optparse 

import pandas as pd
import numpy as np

from crfpp.tagger import tagger
from crfpp.loaddata import loadData
from crfpp.crftools import get_sent_strfeats, get_sent_vecfeats, generate_template, crf_learn, crf_test
from crfpp.evals import get_sent_annoSET, match_anno_pred_result, calculate_F1_Score, logError

def crfpp_train(model, trainSents, Channel_Settings, feat_type = 'str'):
    if 'str' in feat_type.lower():
        get_sent_feats = get_sent_strfeats
    elif 'vec' in feat_type.lower():
        get_sent_feats = get_sent_vecfeats
    
    tmp_dir = model.replace('model', '_tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
        
    if not os.path.exists(model):
        os.makedirs(model)
        
    model_path = model + '/model'

    template_path = model + '/template'
    feats_data_path = tmp_dir + '/feats.txt'
    
    DFtrain = []
    for idx, sent in enumerate(trainSents):
        if idx % 500 == 0:
            print(datetime.now(), idx)
        df = get_sent_feats(sent, Channel_Settings)
        df.loc[len(df)] = np.NaN     ## Trick Here
        DFtrain.append(df)           ## Trick Here
    DFtrain = pd.concat(DFtrain).reset_index(drop=True)
    
    DFtrain.to_csv(feats_data_path, sep = '\t', encoding = 'utf=8', header = False, index = False )

    
    generate_template(input_feats_num = DFtrain.shape[1] - 1, path = template_path) 
    crf_learn(feats_data_path, model_path, template_path  = template_path)
    
    return None


def crfpp_test(model, testSents, Channel_Settings, labels):
    '''
        sents: a list of sents
        This function could be updated in the future for a better performance.
        But currently, just leave it now.
    '''
    pred_entities = []
    anno_entities = []
    log_list      = []
    # here sents are a batch of sents, not necessary to be all the sents
    # this could be chanage, but it will cause some time, so just ignore it.
    for idx, sent in enumerate(testSents):
        if idx % 200 == 0:
            print(datetime.now(), idx)
        # 200/13s
        pred_SET = tagger(model, sent, Channel_Settings = Channel_Settings)
        pred_entities.append(pred_SET)
        
        anno_SET = get_sent_annoSET(sent)
        anno_entities.append(anno_SET)
        
        error = logError(sent, pred_SET, anno_SET)
        log_list.append(error)
    # return anno_entities, pred_entities
    Result = match_anno_pred_result(anno_entities, pred_entities, labels = labels)
    # return Result
    R = calculate_F1_Score(Result, labels)

    LogError = pd.concat(log_list).reset_index(drop = True)
    return R, LogError

def trainModel(model, sentTrain, sentTest, Channel_Settings, labels, cross_idx = 0):
    '''
        sentTrain, sentTest: are featurized already.
    '''
    model = model + '_' + str(cross_idx)
    log_path = model + '/log.csv'
    pfm_path = model + '/performance.csv'
    para   = crfpp_train(model, sentTrain, Channel_Settings)
    R, Err = crfpp_test (model, sentTest,  Channel_Settings, labels)
    Err.to_csv(log_path, index = False, sep = '\t')
    R.to_csv  (pfm_path, index = True,  sep = '\t')
    # generate the error log path
    return R

def train(model, sents, Channel_Settings, labels, cross_num, cross_validation = None, seed = 10):
    '''
        sent is featurized
    '''
    total_pfm_path = model + '_performance.csv'

    if not cross_validation:
        sentTrain, sentTest = loadData(sents, cross_num, seed = seed, cross_validation = cross_validation, cross_idx = 0)
        Performance = trainModel(model, sentTrain, sentTest, Channel_Settings, labels)
        print('\nThe Final Performance is:\n')
        print(Performance)
        Performance.to_csv  (total_pfm_path, index = True,  sep = '\t')
        return Performance
    else:
        L = []
        for cross_idx in range(cross_num):
            sentTrain, sentTest = loadData(sents, cross_num, seed = seed, cross_validation = cross_validation, cross_idx = cross_idx)
            print('For  ', cross_idx, '/', cross_num, "  ====")
            R = trainModel(model, sentTrain, sentTest, Channel_Settings, labels, cross_idx = cross_idx)
            print(R)
            L.append(R)
        Performance = sum(L)/cross_num
        print('\nThe Final Average Performance for', cross_num, 'Cross Validation is:\n')
        print(Performance)
        Performance.to_csv  (total_pfm_path, index = True,  sep = '\t')
        return Performance