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
##############################################################################
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

def generate_template(gram_1 = 20, path = '_tmp/template'):
    L = ['# Unigram\n\n']
    for idx in range(gram_1):
        L.append('U'  + str(idx) + ':%x[0,' + str(idx) + ']\n')
        
    L.append('\n\n')
    L.append('# Bigram\nB')
    with open(path, 'w') as f:
        f.write(''.join(L))
    return ''.join(L)

def crfpp_train(sents, Channel_Settings, model_path, ):
    # create para
    feats_data_path = '_tmp/_crfpp_train_feats.txt'
    template_path   = '_tmp/_template'
    DFtrain = pd.DataFrame()
    for sent in sents:
        df = get_sent_strfeats(sent, Channel_Settings) #
        df.loc[len(df)] = np.NaN     ## Trick Here
        DFtrain = DFtrain.append(df) ## Trick Here
    DFtrain = DFtrain.reset_index(drop=True)
    DFtrain.to_csv(feats_data_path, sep = '\t', encoding = 'utf=8', header = False, index = False )

    generate_template(gram_1 = DFtrain.shape[1] - 1, path = template_path)
    crf_learn(feats_data_path, model_path, template_path  = template_path)
    # return para 
##############################################################################

def read_result(result_path):
    result = pd.read_csv(result_path, sep = '\t', header = None, skip_blank_lines=False)
    # get the last column of the result
    return result.iloc[:,-1].dropna().values

def extractSET(tag_seq, exist_SE = False):
    '''
        SET: start, end, tag
        tag_seq: the hyper field sequence for this sentence
    '''
    if exist_SE:
        tag_seq = tag_seq[1:-1]

    IT = list(zip(range(len(tag_seq)), tag_seq))
    taggedIT = [it for it in IT if it[1]!= 'O']
    
    startIdx = [idx for idx in range(len(taggedIT)) if taggedIT[idx][1][-2:] == '-B']
    startIdx.append(len(taggedIT))

    entitiesList = []
    for i in range(len(startIdx)-1):
        entityAtom = taggedIT[startIdx[i]: startIdx[i+1]]
        # string = ''.join([cit[0] for cit in entityAtom])
        start, end = entityAtom[0][0], entityAtom[-1][0] + 1
        tag = entityAtom[0][1].split('-')[0]
        entitiesList.append((start, end, tag))
    return entitiesList


def get_sent_annoSET(sent, channel = 'annoE', tagScheme = 'BIOES'):
    anno_seq = [i[0] for i in sent.getChannelGrain(channel, tagScheme=tagScheme)]
    anno_SET = extractSET(anno_seq)
    return anno_SET



def tagger(sent, model_path, Channel_Settings = None):
    '''
        basically from crf_test
        sent: a sentence, could be without annotation
    '''
    # 1. get sentence feats
    # hopefully, the model_config is included in model_path

    feats_data_path   = '_tmp/_tagger_feats.txt'
    results_data_path = '_tmp/_tagger_results.txt'

    sentence_with_feats = sent
    df = get_sent_strfeats(sent, Channel_Settings, train = False)
    df.to_csv(feats_data_path, sep = '\t', encoding = 'utf=8', header = False, index = False )
    # 2. save the sentence feats to a file
    
    # 3. tag a sentence
    crf_test(feats_data_path, model_path, results_data_path)

    # 4. read and parse the result to pred_SSET
    # get a tag_seq
    # list of tuple (score, result)
    tag_seq = read_result(results_data_path)
    pred_SET = extractSET(tag_seq)
    return pred_SET


def match_anno_pred_result(anno_entities, pred_entities, label_list = []):
    if type(anno_entities[0]) != list:
        anno_entities = [anno_entities]
        pred_entities = [pred_entities]
        
    name_list = ['E_Anno', 'E_Pred',  'E_Match']
    for eL in label_list:
        name_list.extend([eL + suff for suff in ['_Anno', '_Pred', '_Match']])
    
    statistic_result = []
    
    for idx in range(len(pred_entities)):
        pred = set(pred_entities[idx])
        anno = set(anno_entities[idx])
        d = dict(E_Pred  = len(pred),
                 E_Anno  = len(anno),
                 E_Match = len(pred.intersection(anno)))
        
        for eL in label_list:
            elL = [e for e in pred if eL == e[-1]]
            elA = [e for e in anno if eL == e[-1]]
            elM = set(elA).intersection(set(elL)) ## Union vs Join
            d[eL+'_Pred'] = len(elL)
            d[eL+'_Anno'] = len(elA)
            d[eL+'_Match'] = len(elM)
        
        statistic_result.append(d)
    Result = pd.DataFrame(statistic_result)[name_list]
    return Result

def calculate_F1_Score(Result, label_list):
    Result = Result.sum().to_dict()
    List = []
    entitiesLabel = label_list + ['E']
    # entitiesLabel = ['Sy','Bo', 'Ch', 'Tr', 'Si'] + ['R'] + ['E']
    for eL in entitiesLabel:
        d = dict()
        d['id'] = eL
        for k in Result:
            if eL == k.split('_')[0]:
                d[k.replace(eL + '_','')] = Result[k]
        List.append(d)
    
    R = pd.DataFrame(List)
    R.set_index('id', inplace = True)
    R.index.name = None
    # print(R)
    R['R'] = R['Match']/R['Anno']
    R['P'] = R['Match']/R['Pred']
    R['F1'] = 2*R['R']*R['P']/(R['R'] + R['P'])
    return R[['Anno', 'Pred', 'Match', 'R', 'P', 'F1']]
    

def crfpp_test(sents, Channel_Settings, model_path, label_list):
    '''
        sents: a list of sents
    '''
    pred_entities = []
    anno_entities = []
    # here sents are a batch of sents, not necessary to be all the sents
    for sent in sents:
        pred_SET = tagger(sent, model_path, Channel_Settings)
        pred_entities.append(pred_SET)
        anno_SET = get_sent_annoSET(sent)
        anno_entities.append(anno_SET)
    
    # return anno_entities, pred_entities
    Result = match_anno_pred_result(anno_entities, pred_entities, label_list = label_list)
    # return Result
    R = calculate_F1_Score(Result, label_list)
    return R