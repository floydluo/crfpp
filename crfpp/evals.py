import pandas as pd
import numpy as np
from tabulate import tabulate

###################################################  from IT to SET 

def get_target_field_info(nlptext, Target_Field, fldembed, useStartEnd = False):
    field, para = Target_Field
    # nlptext is important here.
    # channel_anno = 'annoE'
    target_para = {}
    
    if field == 'token':
        # use start and end only works for token
        special_num = 3 if useStartEnd else 1 
        # from fldembed
        Weights = fldembed.weights
        assert 'token' in Weights
        wv = Weights['token']
        GU = wv.GU
        idx2grain = ['</pad>'] + GU[0]
        grain_num = len(idx2grain)
        grain2idx = dict(zip(idx2grain, list(range(grain_num)) ))
        GU = idx2grain, grain2idx
        target_para['GU']  = GU # </pad> in GU, no </unk> in GU
        labels = [] 
        tag_size = len(idx2grain) + special_num # </pad> in GU, and </unk>, </start>, </end> not in GU

    else: 
        # generally, we won't adding start and end into CRF model.
        # even the features are with start and end, it will should be removed before feed into crf
        # it can also be proved that the start and end vectors are useless in sequential labeling.
        GU = nlptext.getGrainVocab(field, **para)
        idx2tag = ['</pad>'] + GU[0]
        tag2idx = dict(zip(range(len(idx2tag)), idx2tag))
        GU = idx2tag, tag2idx
        # </pad> is in GU, no </unk> or other special tags in GU
        tag_size = len(idx2tag)       
        target_para['GU'] = GU
        
        TRANS = nlptext.getTrans(field, **para)
        TRANS = {k: v+1 for k, v in TRANS.items()}
        labels = list(set([i.split('-')[0] for i in idx2tag if '-' in i]))
        labels.sort()
        target_para['TRANS'] = TRANS
        
    # target_para['labels'] = labels
    # target_para['tag_size'] = tag_size
    
    TARGET_FIELD = [field, target_para, labels, tag_size]
    return TARGET_FIELD



def read_target_seq(result_path):
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
    
    startIdx = [idx for idx in range(len(taggedIT)) if taggedIT[idx][1][-2:] == '-B' or taggedIT[idx][1][-2:] == '-S']
    startIdx.append(len(taggedIT))

    entitiesList = []
    for i in range(len(startIdx)-1):
        entityAtom = taggedIT[startIdx[i]: startIdx[i+1]]
        # string = ''.join([cit[0] for cit in entityAtom])
        start, end = entityAtom[0][0], entityAtom[-1][0] + 1
        tag = entityAtom[0][1].split('-')[0]
        entitiesList.append((start, end, tag))
    return entitiesList


def get_sent_annoSET(sent, channel = 'annoE', tagScheme = 'BIO'):
    anno_seq = [i[0] for i in sent.get_grain_str(channel, tagScheme=tagScheme)]
    anno_SET = extractSET(anno_seq)
    return anno_SET
################################################### 



###################################################  compare pred_SET and anno_SET

def match_anno_pred_result(anno_entities, pred_entities, labels = []):
    '''
        anno_entities, pred_entities of a batch of sentences
    '''
    if type(anno_entities[0]) != list:
        anno_entities = [anno_entities]
        pred_entities = [pred_entities]
        
    name_list = ['E@Anno', 'E@Pred',  'E@Match']
    for eL in labels:
        name_list.extend([eL + suff for suff in ['@Anno', '@Pred', '@Match']])
    
    statistic_result = []
    
    for idx in range(len(pred_entities)):
        pred = set(pred_entities[idx])
        anno = set(anno_entities[idx])
        d = {'E@Pred'  : len(pred),
             'E@Anno'  : len(anno),
             'E@Match' : len(pred.intersection(anno))}
        
        for eL in labels:
            elL = [e for e in pred if eL == e[-1]]
            elA = [e for e in anno if eL == e[-1]]
            elM = set(elA).intersection(set(elL)) ## Union vs Join
            d[eL+'@Pred']  = len(elL)
            d[eL+'@Anno']  = len(elA)
            d[eL+'@Match'] = len(elM)
        
        statistic_result.append(d)
    Result = pd.DataFrame(statistic_result)[name_list]
    return Result



def calculate_F1_Score(Result, labels):
    if len(Result.shape) == 1:
        Result = Result.to_dict()
    else:
        Result = Result.sum().to_dict()
    List = []
    entitiesLabel = labels + ['E']
    # entitiesLabel = ['Sy','Bo', 'Ch', 'Tr', 'Si'] + ['R'] + ['E']
    for eL in entitiesLabel:
        d = dict()
        d['id'] = eL
        for k in Result:
            if eL == k.split('@')[0]:
                d[k.replace(eL + '@','')] = Result[k]
        List.append(d)
    
    # print(List)
    R = pd.DataFrame(List)
    R.set_index('id', inplace = True)
    R.index.name = None
    # print(R)
    R['R'] = R['Match']/R['Anno']
    R['P'] = R['Match']/R['Pred']
    R['F1'] = 2*R['R']*R['P']/(R['R'] + R['P'])
    return R[['Anno', 'Pred', 'Match', 'R', 'P', 'F1']]


################################################### log the errors between pred_SET and anno_SET
def matchPaired(L, A, sent):
    start1, end1, e1 = L
    start2, end2, e2 = A
    d = {}
    sentence = sent.sentence.split(' ')
    if set(range(start1, end1+1)).intersection(range(start2, end2+1)):
        idx = set(range(start1, end1+1)).union(range(start2, end2+1))
        # print()
        d['text_part'] = ''.join(sentence[min(idx): max(idx)])
        d['start'] = min(idx)
        d['end' ]  = max(idx) 
        d['pred'] = ''.join(sentence[start1: end1])
        d['pred_en'] = e1
        d['anno'] = ''.join(sentence[start2: end2])
        d['anno_en'] = e2
        d['sent_idx']= sent.Idx # this is important
        return d
    
def matchUnpaired(unpaired, sent, kind):
    d = {}
    sentence = sent.sentence.split(' ')
    d['start'], d['end' ], e = unpaired
    d['text_part'] = ''.join(sentence[d['start']: d['end' ]])
    d['sent_idx']= sent.Idx
    if kind == 'P':
        d['pred'], d['pred_en'] = d['text_part'], e
        d['anno'], d['anno_en'] = None, None
    else:
        d['pred'], d['pred_en'] = None, None
        d['anno'], d['anno_en'] = d['text_part'], e
    return d


def logErrors(sent, anno_enetities, pred_entities):
    log = []
    inter = list(set(pred_entities).intersection(set(anno_enetities)))
    only_pred = [ i for i in pred_entities if i not in inter]   
    only_anno = [ i for i in anno_enetities if i not in inter]
    
    pairedP = []
    pairedA = []
    for L in only_pred:
        for A in only_anno:
            d = matchPaired(L, A, sent)
            if d:
                log.append(d)
                pairedP.append(L)
                pairedA.append(A)
                
    for L in [i for i in only_pred if i not in pairedP]:
        log.append(matchUnpaired(L, sent, 'P'))
        
    for A in [i for i in only_anno if i not in pairedA]:
        log.append(matchUnpaired(A, sent, 'A'))
           
    if len(log) == 0:
        return pd.DataFrame()
    cols = ['sent_idx', 'text_part', 'start', 'end', 'anno', 'anno_en', 'pred', 'pred_en']
    return pd.DataFrame(log)[cols].sort_values('start')

