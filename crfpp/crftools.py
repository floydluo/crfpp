# UPDATE: 2018/03/14, Add concise argument.
# UPDATE: 2018/05/04, For Linxu.

import numpy as np
import pandas as pd
from io import StringIO

console_encoding = 'gb2312'
file_encoding = 'utf-8'


def load_anno(channel_anno, tagScheme, BasicObject):
    # channel_anno = 'annoE'
    GU = BasicObject.getGrainUnique(channel_anno, tagScheme = tagScheme)
    list_annoE = GU[0]
    tag_size = len(list_annoE)
    label_list = list(set([i.split('-')[0] for i in list_annoE if '-' in i]))
    label_list.sort()
    return channel_anno, tagScheme, tag_size, list_annoE, label_list


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


def crf_learn(train_data_path, model_path,
              template_path  = 'template/template01',
              crf_learn_path = 'crftools/crf_learn',
              params = {}):
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
             concise = True, crf_test_path = 'crftools/crf_test'):
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
    features = {}
    # stroke 12 and subcomp 6 are fixed internally.
    for ch, cs in Channel_Settings.items():
        # print(ch)
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
        L.append(pd.DataFrame(feat))
    
    Feats = pd.concat(L, axis = 1)
    return Feats


def get_sent_vecfeats(sent, Channel_Settings, fieldembed, train = True):
    return 


def prepare_sentence_str(BasicObject):
    sentence_path = BasicObject.TokenNum_Dir + '/' + 'Pyramid/_Feat_SENT_Str.crfpp'
    if os.path.isfile(sentence_path):
        with open(sentence_path, 'rb') as handle:
            sents = pickle.load(handle)
        return sents
    else:
        from nlptext.corpus import Corpus
        corpus = Corpus() # this costs time
        sents = [get_sent_strfeats(sent, Total_Settings) for sent in corpus.Sentences]
        with open(sentence_path, 'wb') as handle:
            pickle.dump(sents, handle)
        return sents

def prepare_sentence_vec(BasicObject):
    sentence_path = BasicObject.TokenNum_Dir + '/' + 'Pyramid/_Feat_SENT_Vec.crfpp'
    if os.path.isfile(sentence_path):
        with open(sentence_path, 'rb') as handle:
            sents = pickle.load(handle)
        return sents
    else:
        from nlptext.corpus import Corpus
        corpus = Corpus() # this costs time
        sents = [get_sent_vecfeats(sent, BasicObject.CHANNEL_SETTINGS) for sent in corpus.Sentences]
        with open(sentence_path, 'wb') as handle:
            pickle.dump(sents, handle)
        return sents


def get_dfsent_strfeats(dfsent, Channel_Settings, train = True):
    columns = df.columns
    new_columns = [i for i in columns if i.split('_')[0] in Channel_Settings]
    if not train:
        new_columns = [i for i in new_columns if 'anno' not i in i]
    return dfsent[new_columns]


def get_dfsent_vecfeats(dfsent, Channel_Settings, train = True):
    columns = df.columns
    new_columns = [i for i in columns if i.split('_')[0] in Channel_Settings]
    if not train:
        new_columns = [i for i in new_columns if 'anno' not i in i]
    return dfsent[new_columns] 

##############################################################################


############################################################################## generate template for derivative features
def generate_template(gram_1 = 20, path = '_tmp/template'):
    '''
        from input feats generate derivative feats
    '''
    L = ['# Unigram\n\n']
    for idx in range(gram_1):
        L.append('U'  + str(idx) + ':%x[0,' + str(idx) + ']\n')
        
    L.append('\n\n')
    L.append('# Bigram\nB')
    with open(path, 'w') as f:
        f.write(''.join(L))
    return ''.join(L)
