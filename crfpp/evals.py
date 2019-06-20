import pandas as pd

###################################################  from IT to SET 

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
    anno_seq = [i[0] for i in sent.getChannelGrain(channel, tagScheme=tagScheme)]
    anno_SET = extractSET(anno_seq)
    return anno_SET
################################################### 



###################################################  compare pred_SET and anno_SET

def match_anno_pred_result(anno_entities, pred_entities, labels = []):
    if type(anno_entities[0]) != list:
        anno_entities = [anno_entities]
        pred_entities = [pred_entities]
        
    name_list = ['E_Anno', 'E_Pred',  'E_Match']
    for eL in labels:
        name_list.extend([eL + suff for suff in ['_Anno', '_Pred', '_Match']])
    
    statistic_result = []
    
    for idx in range(len(pred_entities)):
        pred = set(pred_entities[idx])
        anno = set(anno_entities[idx])
        d = dict(E_Pred  = len(pred),
                 E_Anno  = len(anno),
                 E_Match = len(pred.intersection(anno)))
        
        for eL in labels:
            elL = [e for e in pred if eL == e[-1]]
            elA = [e for e in anno if eL == e[-1]]
            elM = set(elA).intersection(set(elL)) ## Union vs Join
            d[eL+'_Pred'] = len(elL)
            d[eL+'_Anno'] = len(elA)
            d[eL+'_Match'] = len(elM)
        
        statistic_result.append(d)
    Result = pd.DataFrame(statistic_result)[name_list]
    return Result


def calculate_F1_Score(Result, labels):
    Result = Result.sum().to_dict()
    List = []
    entitiesLabel = labels + ['E']
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
###################################################



################################################### log the errors between pred_SET and anno_SET

def matchPaired(L, A, sent):
    start1, end1, e1 = L
    start2, end2, e2 = A
    d = {}
    sentence = sent.sentence
    if set(range(start1, end1+1)).intersection(range(start2, end2+1)):
        idx = set(range(start1, end1+1)).union(range(start2, end2+1))
        # print()
        d['text_part'] = sent.sentence[min(idx): max(idx) + 1]
        d['start'] = min(idx)
        d['end' ]  = max(idx) 
        d['pred'] = sent.sentence[start1: end1]
        d['pred_en'] = e1
        d['anno'] = sent.sentence[start2: end2]
        d['anno_en'] = e2
        d['sent_idx']= sent.Idx # this is important
        return d
    
def matchUnpaired(unpaired, sent, kind):
    d = {}
    sentence = sent.sentence
    d['start'], d['end' ], e = unpaired
    d['text_part'] = sentence[d['start']: d['end' ]]
    d['sent_idx']= sent.Idx
    if kind == 'P':
        d['pred'], d['pred_en'] = d['text_part'], e
        d['anno'], d['anno_en'] = None, None
    else:
        d['pred'], d['pred_en'] = None, None
        d['anno'], d['anno_en'] = d['text_part'], e
    return d


def logError(sent, pred_entities, anno_enetities):
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

###################################################