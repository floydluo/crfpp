# UPDATE: 2018/03/14, Add concise argument.
# UPDATE: 2018/05/04, For Linxu.
import os
import pickle
import numpy as np
import pandas as pd
from io import StringIO

console_encoding = 'gb2312'
file_encoding = 'utf-8'


def load_anno(BasicObject, anno_field):
    # channel_anno = 'annoE'
    Channel_Settings = BasicObject.CHANNEL_SETTINGS
    tagScheme = Channel_Settings[anno_field]['tagScheme']
    GU = BasicObject.getGrainUnique(anno_field, tagScheme = tagScheme)
    tags = GU[0]
    tag_size = len(tags)
    labels = list(set([i.split('-')[0] for i in tags if '-' in i]))
    labels.sort()
    return Channel_Settings, tagScheme, labels, tags, tag_size

def get_model_name(BasicObject, Channel_Settings):
    return BasicObject.TokenNum_Dir.replace('data/', 'model/') + '/' + "_".join([getChannelName(ch, style = 'abbr') 
                                            for ch in Channel_Settings if 'anno' not in ch])

def dict2list(paramdict):
    resultlist = []
    for k, v in paramdict.items():
        resultlist.append(k)
        if v: resultlist.append(v)
    return resultlist

def shell_invoke(args, sinput = None, soutput = None):
    import subprocess
    if sinput and soutput:
        p = subprocess.Popen(args, stdin = sinput, stdout= soutput)
    elif sinput:
        p = subprocess.Popen(args, stdin=sinput, stdout=subprocess.PIPE)
    elif soutput:
        p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=soutput)
    else:
        p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    result = p.communicate()
    for robj in result:
        if robj:
            print(robj.decode(console_encoding))
    return None


learn_params  = {'-f': '2',
                 '-c': '5.0',
                 '-t': None}

def crf_learn(train_data_path, model_path,
              template_path  = 'template/template01',
              crf_learn_path = 'source/crf_learn',
              params = learn_params):
    '''
    Use train data from `train_data_path` to learn a model and save the model to `model_path`
    You may need to specify template_path or crf_learn_path
    '''
    part_args = []
    if params:
        part_args += dict2list(params)
    part_args += [template_path, train_data_path, model_path]
    try:
        shell_invoke([crf_learn_path] + part_args)
    except:
        shell_invoke(['crf_learn'] + part_args)

def crf_test(test_data_path, model_path, result_path, 
             concise = True, crf_test_path = 'source/crf_test'):
    '''
    Use test data from `test_data_path` and the model from `model_path` to save the result in `result_path`
    You may need to specify the concise or crf_test_path
    '''
    if not concise: 
        part_args = ['-v', '2', '-m', model_path, test_data_path]
        with open(result_path, 'w') as fh_write:
            try:
                shell_invoke([crf_test_path] + part_args, soutput = fh_write)
            except:
                shell_invoke(['crf_test'] + part_args, soutput = fh_write)
    else:
        part_args = ['-m', model_path, test_data_path]
        
        with open(result_path, 'w') as fh_write:
            try:
                shell_invoke([crf_test_path] + part_args, soutput = fh_write)
            except:
                shell_invoke(['crf_test'] + part_args, soutput = fh_write)


# Dev in 2019/06/14


############################################################################## prepare feats

Total_Settings = {
    'token': {'Max_Ngram': 1},
    'char': {'Max_Ngram': 1, 'end_grain': False},
    'basic': {'Max_Ngram': 1, 'end_grain': False},
    'radical': {'Max_Ngram': 1, 'end_grain': False},
    'subcomp': {'Max_Ngram': 1, 'end_grain': False},
    'stroke': {'Max_Ngram': 1, 'end_grain': False},
    'pos': {'Max_Ngram': 1, 'end_grain': False,   'tagScheme': 'BIOE'},
    'annoE': {'Max_Ngram': 1, 'end_grain': False, 'tagScheme': 'BIOE'}
}

def get_sent_strfeats(sent, Channel_Settings, train = True):
    '''
        sent is a nlptex.sentence object
        return a pandas dataframe
    '''
    try:
        df = sent.feats
        columns = df.columns
        new_columns = [i for i in columns if i.split('_')[0] in Channel_Settings]
        if not train:
            new_columns = [i for i in new_columns if 'anno' not in i]
        Feats = df.loc[new_columns]
        return Feats
    except:
        features = {}
        # stroke 12 and subcomp 6 are fixed internally.
        for ch, cs in Channel_Settings.items():
            if 'anno' in ch and not train:
                continue
            feature = sent.getChannelGrain(ch)
            # this will cost a lot of time
            if ch == 'stroke':
                max_leng = 12
                feature2 = []
                for token_feat in feature:
                    if len(token_feat) <= max_leng:
                        feature2.append(token_feat + ['</'] * (max_leng - len(token_feat))) 
                    else:
                        feature2.append(token_feat[:max_leng])
                feature = feature2
            elif ch == 'subcomp':
                max_leng = 6
                feature2 = []
                for token_feat in feature:
                    if len(token_feat) <= max_leng:
                        feature2.append(token_feat + ['</'] * (max_leng - len(token_feat)))
                    else:
                        feature2.append(token_feat[:max_leng])
                feature = feature2 
            features[ch] = feature
            # print(feature)
        L = []
        for ch, feat in features.items():
            df = pd.DataFrame(feat)
            df.columns = [ch + '_' + str(i) for i in range(df.shape[1])]
            L.append(df)
        
        Feats = pd.concat(L, axis = 1)
        return Feats

def get_sent_vecfeats(sent, Channel_Settings, fieldembed, train = True):
    return 

def featurize_nlptext_sentences(BasicObject, feat_type = 'str', fieldembed = None):
    if 'str' in feat_type.lower():
        sentence_path = BasicObject.TokenNum_Dir + '/' + 'Pyramid/_Feat_SENT_Str.crfpp'
        get_sent_feats = get_sent_strfeats
    elif 'vec' in feat_type.lower():
        sentence_path = BasicObject.TokenNum_Dir + '/' + 'Pyramid/_Feat_SENT_Vec.crfpp'
        get_sent_feats = get_sent_vecfeats
    else:
        print('Error! Currently there is no such a feature type')
        
    if os.path.isfile(sentence_path):

        with open(sentence_path, 'rb') as handle:
            sents = pickle.load(handle)
        print('Load Featurized Sentences from:')
        print('\t', sentence_path)
        return sents
    else:
        from nlptext.corpus import Corpus
        corpus = Corpus() # this costs time
        sents = []
        for sent in corpus.Sentences:
            sent.feats = get_sent_feats(sent, Total_Settings)
            sents.append(sent)
        with open(sentence_path, 'wb') as handle:
            pickle.dump(sents, handle)
        print('Save Featurized Sentences to:')
        print('\t', sentence_path)
        return sents

##############################################################################


############################################################################## generate template for derivative features

individual_para = {
    1: 3,
    2: 2,
    3: 1,
}

def generate_template(input_feats_num = 5, individual_para = individual_para, path = None):
    '''
     this still needs more consideration
    '''
    if not path:
        path = '_template'
        
    L = ['# Unigram\n\n']
    fld_idx = 0
    for feat_idx in range(input_feats_num):
        for gram_num, window_size in individual_para.items():
            if gram_num == 1:
                for token_i in range(-window_size, window_size + 1):
                    L.append('U{}:%x[{},{}]\n'.format(str(fld_idx), str(token_i), str(feat_idx)))
                    fld_idx = fld_idx + 1
            if gram_num == 2 and feat_idx <= 5:
                for token_i in range(-window_size, window_size + 1):
                    L.append('U{}:%x[{},{}]/%x[{},{}]\n'.format(str(fld_idx), 
                                                                str(token_i), str(feat_idx), 
                                                                str(token_i + 1), str(feat_idx)))
                    fld_idx = fld_idx + 1
                    
            if gram_num == 3 and feat_idx <= 5:
                for token_i in range(-window_size, window_size + 1):
                    L.append('U{}:%x[{},{}]/%x[{},{}]/%x[{},{}]\n'.format(str(fld_idx), 
                                                                          str(token_i - 1), str(feat_idx), 
                                                                          str(token_i), str(feat_idx), 
                                                                          str(token_i + 1), str(feat_idx)))
                    fld_idx = fld_idx + 1
    
    
    L.append('\n\n# Bigram\nB')
    
    with open(path, 'w') as f:
        f.write(''.join(L))
        
    return ''.join(L)
