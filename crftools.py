
console_encoding = 'gb2312'
file_encoding = 'utf-8'


# UPDATE: 2018/03/14, Add concise argument.
# UPDATE: 2018/05/04, For Linxu.

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
             concise = False, crf_test_path = 'crftools/crf_test'):
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

tmp_result_path = 'tmp.txt'
def tagger(sentence_with_feats, model_path):
    '''basically from crf_test'''
    # 1. get sentence feats
    sentence_with_feats = None

    # 2. save the sentence feats to a file
    feats_data_path   = 'tmp/_feats.txt'
    results_data_path = 'tmp/_results.txt'

    # 3. tag a sentence
    crf_test(feats_data_path, model_path, results_data_path)

    # 4. read and parse the result
    pred_SSET = []

    anno_SSET = []

    return pred_SSET