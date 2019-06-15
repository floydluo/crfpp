import pandas as pd
from splitresult import splitResult

#resultPath = tmpOutput

def match(L, A, cct):
    t1, start1, end1, e1 = L
    t2, start2, end2, e2 = A
    d = {}
    if set(range(start1, end1+1)).intersection(range(start2, end2+1)):
        idx = set(range(start1, end1+1)).union(range(start2, end2+1))
        # print()
        d['Text'] = cct._text[min(idx): max(idx) + 1]
        d['T_start'] = min(idx)
        d['T_end' ]  = max(idx) 
        d['Learned'] = t1
        d['LearnET'] = e1
        d['Annoted'] = t2
        d['AnnotET'] = e2
        d['FilePath']= cct.annotedFilePath
        return d
    
    
def matchUpair(unpaired, cct, kind):
    d = {}
    d['Text'], d['T_start'], d['T_end' ], e = unpaired
    d['FilePath']= cct.annotedFilePath
    if kind == 'L':
        d['Learned'], d['LearnET'] = d['Text'], e
        d['Annoted'], d['AnnotET'] = None, None
    else:
        d['Learned'], d['LearnET'] = None, None
        d['Annoted'], d['AnnotET'] = d['Text'], e
    return d


def logError(cct):
    log = []
    inter = list(set(cct.learnedEntities).intersection(set(cct.annotedEntities)))
    OnlyLearned = [ i for i in cct.learnedEntities if i not in inter]   
    OnlyAnnoted = [ i for i in cct.annotedEntities if i not in inter]
    
    pairedL = []
    pairedA = []
    for L in OnlyLearned:
        for A in OnlyAnnoted:
            d = match(L, A, cct)
            if d:
                log.append(d)
                pairedL.append(L)
                pairedA.append(A)
                
    for L in [i for i in OnlyLearned if i not in pairedL]:
        log.append(matchUpair(L, cct, 'L'))
        
    for A in [i for i in OnlyAnnoted if i not in pairedA]:
        log.append(matchUpair(A, cct, 'A'))
           
    if len(log) == 0:
        return pd.DataFrame()
    cols = ['FilePath', 'Text', 'T_start', 'T_end', 'Annoted', 'AnnotET', 'Learned', 'LearnET']
    return pd.DataFrame(log)[cols].sort_values('T_start')

