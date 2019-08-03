import os
import time
import codecs
import optparse 
import pickle

import numpy as np
import pandas as pd

from crfpp.crftools import get_sent_strfeats, crf_test
from crfpp.evals import read_target_seq, extractSET

def tagger(model, sent, Channel_Settings = None):
    '''
        basically from crf_test
        sent: a sentence, could be without annotation
    '''
    # model = model + '_' + str(cross_idx)
    if not Channel_Settings:
        with open(model + '/para.p', 'rb') as f:
            Channel_Settings = pickle.load(f)
    # 1. get sentence feats
    # hopefully, the model_config is included in model_path
    feats_data_path   = '_tmp/_tagger_feats.txt'
    results_data_path = '_tmp/_tagger_results.txt'
    model_path = model + '/model'

    # get Channel_Settings
    # get use sent strfeats or sent vecfeats settings
    df = get_sent_strfeats(sent, Channel_Settings, train = False)
    df.to_csv(feats_data_path, sep = '\t', encoding = 'utf=8', header = False, index = False )
    # 2. save the sentence feats to a file
    
    # 3. tag a sentence
    crf_test(feats_data_path, model_path, results_data_path)

    # 4. read and parse the result to pred_SSET
    # get a tag_seq
    # list of tuple (score, result)
    tag_seq  = read_target_seq(results_data_path)
    pred_SET = extractSET(tag_seq)
    return pred_SET

