import optparse 
import pickle
from pprint import pprint

from nlptext.base import BasicObject
from nlptext.corpus import Sentence

from crfpp.crftools import load_anno, get_model_name
from crfpp.train import crfpp_train, crfpp_test
from crfpp.loaddata import get_train_test_valid


########### ResumeNER ###########
from pprint import pprint
from nlptext.base import BasicObject


Data_Dir = 'data/MedPos/char'
BasicObject.INIT_FROM_PICKLE(Data_Dir)  


# purly one hot 
FIELD_SETTINGS_TEMPLATE = {
    # CTX_IND
    'token':   {'use': True, 'Max_Ngram': 1,}, # always the char-level token
    'basic':   {'use': False, 'Max_Ngram': 1, 'end_grain': False},
    'medical': {'use': False, 'Max_Ngram': 1, 'end_grain': False},
    'radical': {'use': False, 'Max_Ngram': 1, 'end_grain': False},
    'subcomp': {'use': False,'Max_Ngram': 1, 'end_grain': False},
    'stroke':  {'use': False,'Max_Ngram': 1, 'end_grain': False},
    # CTX_DEP
    'pos':     {'use': False, 'tagScheme': 'BIOES',},
    # ANNO
    'annoE':   {'use': True, 'tagScheme': 'BIOES',},
}


# FIELD_SETTINGS_TEMPLATE = {
#     # CTX_IND
#     'token':   {'use': True,'Max_Ngram': 1,}, # always the char-level token
#     # CTX_DEP
#     'pos':     {'use': True, 'tagScheme': 'BIOE',},
#     # ANNO
#     'annoE':   {'use': True, 'tagScheme': 'BIOE',},
# }


BasicObject.BUILD_GV_LKP(FIELD_SETTINGS_TEMPLATE)


if __name__ == '__main__':
    parser = optparse.OptionParser()

    # parser.add_option(
    #     "-v", "--model", default='1abdp',
    #     help="Model name"
    # )

    ###########################################

    anno_field = 'annoE'
    Channel_Settings, tagScheme, labels, tags, tag_size = load_anno(BasicObject, anno_field)


    model = get_model_name(BasicObject, Channel_Settings)
    print('The model path is:')
    print('\t', model)

    
    para_path = model + '/para.p'
    with open(para_path, 'wb') as handle:
        pickle.dump(Channel_Settings, handle )

    ###########################################

    # corpus = Corpus()
    # sents = corpus.Sentences
    total_sent_num = BasicObject.SENT['length']
    train_sent_idx, test_sent_idx, valid_sent_idx = get_train_test_valid(total_sent_num, prop = 0.8, seed = 10)
    train_sents = [Sentence(i) for i in train_sent_idx]
    test_sents  = [Sentence(i) for i in test_sent_idx]
    valid_sents = [Sentence(i) for i in valid_sent_idx]


    # model = model + '_' + str(cross_idx)
    para   = crfpp_train(model, train_sents, Channel_Settings)

    log_path = model + '/valid_log.csv'
    pfm_path = model + '/valid_performance.csv'

    R, Err = crfpp_test(model, valid_sents,  Channel_Settings, labels)
    Err.to_csv(log_path, index = False, sep = '\t')
    R.to_csv  (pfm_path, index = True,  sep = '\t')

    print('\nFor Valid:\n')
    print(R)


    log_path = model + '/test_log.csv'
    pfm_path = model + '/test_performance.csv'
    
    R, Err = crfpp_test(model, test_sents,  Channel_Settings, labels)
    Err.to_csv(log_path, index = False, sep = '\t')
    R.to_csv  (pfm_path, index = True,  sep = '\t')


    print('\nFor Test:\n')
    print(R)
