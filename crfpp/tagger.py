import os
import time
import codecs
import optparse 
import pickle

import numpy as np
import pandas as pd

from crftools import get_sent_strfeats, crf_test, read_target_seq, extractSET


def tagger(sent, model_path):
    '''
        basically from crf_test
        sent: a sentence, could be without annotation
    '''
    # 1. get sentence feats
    # hopefully, the model_config is included in model_path
    feats_data_path   = '_tmp/_tagger_feats.txt'
    results_data_path = '_tmp/_tagger_results.txt'

    # get Channel_Settings
    # get use sent strfeats or sent vecfeats settings

    # we use get_sent_strfeats instead of get_dfsent_strfeats
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

if __name__ == '__main__':
    parser = optparse.OptionParser()

    parser.add_option(
        "-m", "--model", default='1abdp',
        help="Model name"
    )

    parser.add_option(
        "-i", "--input", default="",
        help="Input file location"
    )

    parser.add_option(
        "-o", "--output", default="",
        help="Output file location"
    )

    parser.add_option(
        "-b", "--batch", default="ccks",
        help="batch name"
    )
    opts = parser.parse_args()[0]
    tagger(model, inputPathFile, outputPathFile, batch)