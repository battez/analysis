'''
try out a gensim doc2vec model train on tweets 
'''

# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec, Word2Vec

# numpy
import numpy

# random
from random import shuffle

# classifier
from sklearn.linear_model import LogisticRegression


import sys
import pymongo
TWITDIR = 'U:\Documents\Project\scrape'
CURR_PLATFORM = sys.platform
if CURR_PLATFORM != 'linux':
    TWITDIR = 'U:\Documents\Project\demoapptwitter'
    SCRAPEDIR = 'U:\Documents\Project\scrape'
else:
    TWITDIR = '/home/luke/programming/'
    SCRAPEDIR = '/home/luke/programming/scraping'#FIXME:

sys.path.insert(0, TWITDIR)
sys.path.insert(0, SCRAPEDIR)

# get some handy functions 
import jlpb
import prepare

# get training and test data from mongodb:
if CURR_PLATFORM != 'linux':
    dbc = jlpb.get_dbc('Twitter', 'sampledate23_500')
    dbun = jlpb.get_dbc('Twitter', 'buffer50k')
    dbt = jlpb.get_dbc('Twitter', 'buffer10k')


class LabeledLineSentence(object):
    '''
    credit: http://linanqiu.github.io/2015/10/07/word2vec-sentiment/
    this class taken from URL above, to assist with using Doc2Vec Sentence Labels
    and multiple files as typically used to train test models etc.
    '''
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences



def quick_label_tweets():
    '''
    run through a mongodb collection of tweets, prompting -
    asking user to label the tweet before moving to next one.
    '''
    docs = dbt.find()
    count = 0

    while docs:

        response = input('text: ' + jlpb.uprint(doc['text']) + \
            "\n user:" + jlpb.uprint(doc['user']['screen_name']) + \
            "\n place:" + jlpb.uprint(doc['place']['full_name']))
        count += 1
        print('for ' + count + ', you said ', response.upper())

        # update db record
    
    print('docs complete')


if __name__ == '__main__':   

    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
     level=logging.INFO)

    # What are we doing with this run of the script? 
    # 
    WRITE_OUT = False
    CREATE_MODEL = True # set false to load the model with chosen parameters

    fout = 'model_d2v.d2v' #output filename suffix

    # train and test sets set up:
    files = ['train_pos_frm500.txt','train_neg_frm500.txt',\
     'test_pos_frm500.txt', 'test_neg_frm500.txt', 'unlabelled.txt']
    
    ftrainpos = open(files[0], 'a', encoding='utf-8') 
    ftrainneg = open(files[1], 'a', encoding='utf-8')
    ftestpos = open(files[2], 'a', encoding='utf-8') 
    ftestneg = open(files[3], 'a', encoding='utf-8')
    funlabelled = open(files[4], 'a', encoding='utf-8')

    # split amounts
    ntest = 50
    ntrain = 450

    if WRITE_OUT:

        # take out Test set first:
        docs = dbc.find()

        for doc in docs[:ntest]:

            # normalise tweet text
            text = prepare.normalise_tweet(doc['text'], unicode_replace=False)
            
            # write to file with label
            # write to diff file depending on t_Class label
            if doc['t_class']:
                ftestpos.write(text + '\n')
            else:
                ftestneg.write(text + '\n')
        
        del docs
        
        # Now write out the Train set (two classes):
        docs = dbc.find()
        for doc in docs[ntest:]:

            # normalise tweet text
            text = prepare.normalise_tweet(doc['text'], unicode_replace=False)
            
            # write to file with label
            # write to diff file depending on t_Class label
            if doc['t_class']:
                ftrainpos.write(text + '\n')
            else:
                ftrainneg.write(text + '\n')

        del docs

        # Then write out the Unlabelled:
        docs = dbun.find()
        for doc in docs:

            # normalise tweet text
            text = prepare.normalise_tweet(doc['text'], unicode_replace=False)
            funlabelled.write(text + '\n')
        
        del docs

    #
    # PARAMS for Word2Vec & Doc2Vec =======================================
    #
    #
    epochs = 18
    vec_length = 92 # rule thumb, sqrt of vocab.
    window=8
    vec_type = 'doc2vec'
    negsample=6
    sample=1e-4

    if CREATE_MODEL:
        
        sources = { files[0]:'TRAIN_POS',\
                    files[1]:'TRAIN_NEG',\
                    files[2]:'TEST_POS',\
                    files[3]:'TEST_NEG',\
                    files[4]:'TRAIN_UNS'}

        sentences = LabeledLineSentence(sources)

        if vec_type == 'doc2vec':
            # sample=1e-4, negative=5, 
            model = Doc2Vec(min_count=1, window=6, size=vec_length, workers=8, negative=negsample, sample=sample)
            model.build_vocab(sentences.to_array())
            # more is better but longer... ~ 20 ideal
            for epoch in range(epochs):
                model.train(sentences.sentences_perm())
        else:
            model = Word2Vec(min_count=10, window=window, size=vec_length, workers=8)
            model.build_vocab(sentences)
            # more is better but longer... ~ 20 ideal
            for epoch in range(epochs):
                '''
                in every training epoch, the set of sentences input to the model 
                gets randomised
                ''' 
                model.train(sentences.sentences_perm())
       

        

        model.save('./sample_neg' + str(negsample) + '_' + \
            str(epochs) + '_' + str(vec_length) + fout)

    if vec_type == 'doc2vec':
        model = Doc2Vec.load('./sample_neg' + str(negsample) + '_' + str(epochs) + '_' + str(vec_length) + fout)
    else:
        model = Word2Vec.load('./' + str(epochs) + '_' + str(vec_length) + fout)

    

    f = open('d2v_results.txt','a')
    vocab = print('Vocab. length: ', str(len(list(model.vocab.keys()))), file=f)

    print('\n')
    print('Vector size:  ', vec_length, 'Type:', vec_type, 'Window: ', window, file=f)
    print(' (epochs =', str(epochs), ')', file=f)
    print('========================================================')
    print('\n\ntest algebra:', file=f)
    pos = ['brexit']
    neg = ['leave', 'voteleave']
    print('debug numpy array')
    print('sentence vector: numpy', model.syn0.shape) # vocab * vec_length 
    print(type(model.syn0))
    print(model.syn0[0,0])

    print(model.most_similar('bexleyheath'))
    print(model.most_similar('surbiton'))
    print(model.most_similar('catford'))
    print(model.most_similar('love'))
    # do a test run on docs:
    doctest = ['brexit','I','voted','just','now'] 
    res = model.infer_vector(doctest)
    print(res)
    doctest = ['flooding','raining','this','morning','london'] 
    res = model.infer_vector(doctest)
    print(res)
   
    #   negative=neg,topn=5), file=f)
    # print('+'.join(pos) + ' - (' + '+'.join(neg) + ')', file=f)
    # jlpb.uprint(model.most_similar(positive=pos,\
    #   negative=neg,topn=5), file=f)
    # jlpb.uprint(model.most_similar(positive=['voteleave','leave'],\
    #   topn=5), file=f)
    # jlpb.uprint(model.most_similar(positive=['voteremain','remain'],\
    #   topn=5), file=f)
    # jlpb.uprint(model.most_similar(positive=['voting','euref'],\
    #   topn=5), file=f)

    # Only works in Word2Vec objects:
    # jlpb.uprint(model.similar_by_vector([1,2]))
    # jlpb.uprint(model.n_similarity(['football','tv'],['rain','storm']), file=f) 
    # items = ['brexit','storm','flooding','weather','farage','immigrants']
    # for item in items:
    #     jlpb.uprint('nsimilar: ' + item + ' - ',  model.similar_by_word(list(item)), file=f) 


    from tabulate import tabulate

    STYLETABLE = 'psql'
    headers = ['Odd one out?', 'Answer']
    terms = [['thunder','lightning','weather','euref'], ['remain','southernrail','brexit','ukip'],\
    ['rain','train','work','morning']]
    answer = []
    for term in terms:
        answer.append([term, model.doesnt_match(term)])
    jlpb.uprint(u''+tabulate(answer, headers=headers, tablefmt=STYLETABLE), file=f)
    
    terms = ['brexit', 'flooding','flood', 'weather', 'rain','love','friend','cat','dog']
    mostlike = []
    headers = ['Keyword', 'Most similar terms (cosine sim.)']
    for term in terms:
        mostlike.append([term, model.most_similar(term)])
    jlpb.uprint(tabulate(mostlike, headers=headers, tablefmt=STYLETABLE), file=f)

    comparisons = [('bexleyheath','flooding'), ('rain', 'flooding'), ('london','flooding'), ('essex','flooding'), ('floods','storms'), \
     ('thunderstorm', 'lightning'), ('weather', 'delays'), ('weather','trains'), ('boris','cameron'),\
      ('remain','brexit'), ('leave','brexit'), ('flood', 'brexit'),('voteleave', 'voteremain'), ('eu','brexit'),('love','flooding')]
    similarity = []
    headers = ['Cosine Similarity Between', 'Value']
    for item in comparisons:
        similarity.append( [','.join(item), str(model.similarity(item[0], item[1])) ])
    jlpb.uprint(tabulate(similarity, headers=headers, tablefmt=STYLETABLE), file=f)

    
