# UPDATE: 2018/03/14, Add concise argument.
# UPDATE: 2018/05/04, For Linxu.


import pandas as pd
from io import StringIO


console_encoding = 'gb2312'
file_encoding = 'utf-8'

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
              crf_learn_path = 'crftools/crf_learn'):
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


def crfpp_train(sents, model_path, template_path = 'template/template01'):
    # create para
    para = []
    # create template_path
    # create sentence input feats
    sentence_with_feats = sent

    feats_data_path   = 'tmp/_crfpp_train_feats.txt'

    crf_learn(feats_data_path, model_path,
              template_path  = template_path)
    
    return para 


def read_result(result_path):
    result = pd.read_csv(result_path, sep = '\t', header = None, skip_blank_lines=False)
    # get the last column of the result
    return tag_seq[-1].values


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



def tagger(sent, model_path, para = None):
    '''
        basically from crf_test
        sent: a sentence, could be without annotation
    '''
    # 0. get para
    if not para:
        para = [] 
    # 1. get sentence feats
    # hopefully, the model_config is included in model_path
    sentence_with_feats = sent

    # 2. save the sentence feats to a file
    feats_data_path   = 'tmp/_tagger_feats.txt'
    results_data_path = 'tmp/_tagger_results.txt'

    # 3. tag a sentence
    crf_test(feats_data_path, model_path, results_data_path)

    # 4. read and parse the result to pred_SSET
    # get a tag_seq
    # list of tuple (score, result)
    tag_seq = read_result(result_path)
    pred_SET = extractSET(tag_seq)

    return pred_SSET

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
    Result = Result.to_dict()
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

    R['R'] = R['Match']/R['Anno']
    R['P'] = R['Match']/R['Pred']
    R['F1'] = 2*R['R']*R['P']/(R['R'] + R['P'])
    return R[['Anno', 'Pred', 'Match', 'R', 'P', 'F1']]

def crfpp_test(sents, model_path, para = None):
    '''
        sents: a list of sents
    '''
    # 0. get para
    if not para:
        para = [] 

    label_list = [] # get the label list
    pred_entities = []
    anno_entities = []
    # here sents are a batch of sents, not necessary to be all the sents
    for sent in sents:
        pred_SET = tagger(sent, model_path)
        pred_entities.append(pred_SET)
        anno_SET = sent.get_SSET() # to specify
        anno_entities.append(anno_SET)

    Result = match_anno_pred_result(anno_entities, pred_entities, label_list = label_list)
    R = calculate_F1_Score(Result, label_list)
    return R