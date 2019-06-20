from pprint import pprint

from nlptext.base import BasicObject
from nlptext.corpus import Corpus

from crfpp.crftools import load_anno, get_model_name
from crfpp.train import train


########### ResumeNER ###########
CORPUSPath = 'corpus/ResumeNER/'
corpusFileIden = '.bmes'
textType   = 'block'
Text2SentMethod  = 're'
Sent2TokenMethod = 'iter'
TOKENLevel = 'char'
anno = 'embed'
annoKW = {}

BasicObject.INIT(CORPUSPath, corpusFileIden, textType, 
                 Text2SentMethod, Sent2TokenMethod, TOKENLevel, 
                 anno, annoKW)

CHANNEL_SETTINGS_TEMPLATE = {
    # CTX_IND
    'token':   {'use': True,'Max_Ngram': 1,}, # always the char-level token
    'basic':   {'use': True,'Max_Ngram': 1, 'end_grain': False},
    'medical': {'use': False,'Max_Ngram': 1, 'end_grain': False},
    'radical': {'use': False,'Max_Ngram': 1, 'end_grain': False},
    'subcomp': {'use': False,'Max_Ngram': 1, 'end_grain': False},
    'stroke':  {'use': False,'Max_Ngram': 1, 'end_grain': False},
    # CTX_DEP
    'pos':     {'use': False, 'tagScheme': 'BIO',},
    # ANNO
    'annoE':   {'use': True, 'tagScheme': 'BIO',},
}

BasicObject.BUILD_GRAIN_UNI_AND_LOOKUP(CHANNEL_SETTINGS_TEMPLATE=CHANNEL_SETTINGS_TEMPLATE)


if __name__ == '__main__':
    parser = optparse.OptionParser()

    parser.add_option(
        "-v", "--model", default='1abdp',
        help="Model name"
    )

    ###########################################

    anno_field = 'annoE'
    Channel_Settings, tagScheme, labels, tags, tag_size = load_anno(BasicObject, anno_field)

    model = get_model_name(BasicObject, Channel_Settings)
    print(model)

    ###########################################

    corpus = Corpus()
    sents = corpus.Sentences

    cross_num = 4
    cross_validation = True
    seed = 10
    train(model, sents, Channel_Settings, labels, cross_num, cross_validation = cross_validation, seed = seed)

    ###########################################
