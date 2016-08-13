'''
try out a gensim doc2vec model train on tweets 
'''

# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

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


class LabeledLineSentence(object):
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


if __name__ == '__main__':   


    WRITE_OUT = False
    CREATE_MODEL = False
    fout = 'model_d2v.d2v'

    # train and test sets set up:
    files = ['train_pos_frm500.txt','train_neg_frm500.txt', 'test_pos_frm500.txt', 'test_neg_frm500.txt', 'unlabelled.txt']
    
    ftrainpos = open(files[0], 'a', encoding='utf-8') 
    ftrainneg = open(files[1], 'a', encoding='utf-8')
    ftestpos = open(files[2], 'a', encoding='utf-8') 
    ftestneg = open(files[3], 'a', encoding='utf-8')
    funlabelled = open(files[4], 'a', encoding='utf-8')

    # split amounts
    ntest = 50
    ntrain = 450

    if WRITE_OUT:
        docs = dbc.find()

        # take out test set first:
        for doc in docs[:ntest]:

            # normalise tweet text
            text = prepare.normalise_tweet(doc['text'], unicode_replace=False)
            # jlpb.uprint(text)
            
            # write to file with label
            # write to diff file depending on t_Class label
            if doc['t_class']:
                ftestpos.write(text + '\n')
            else:
                ftestneg.write(text + '\n')
        
        del docs
        
        docs = dbc.find()
        # then write out the train set (two classes):
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

        # then write out the unlabelled:
        docs = dbun.find()
        for doc in docs:

            # normalise tweet text
            text = prepare.normalise_tweet(doc['text'], unicode_replace=False)
            
            funlabelled.write(text + '\n')
        
        del docs

    if CREATE_MODEL:
        epochs = 20
        sources = { files[0]:'TRAIN_POS',\
                    files[1]:'TRAIN_NEG',\
                    files[2]:'TEST_POS',\
                    files[3]:'TEST_NEG',\
                    files[4]:'TRAIN_UNS'}

        sentences = LabeledLineSentence(sources)

        model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=3)

        # using mostly the buffer data from that day.
        model.build_vocab(sentences.to_array())

        # more is better but longer... ~ 20 ideal
        for epoch in range(epochs):
            model.train(sentences.sentences_perm())

        model.save('./' + str(epochs) + fout)

    model = Doc2Vec.load('./' + str(20) + fout)

    print('test algebra')
    print(model.most_similar(positive=['eu', 'out'], negative=['remain'], topn=10))
    print('no match or odd one out')
    
    from tabulate import tabulate

    STYLETABLE = 'psql'
    headers = ['Odd one out?', 'Answer']
    terms = [['thunder','lightning','weather','euref'], ['remain','southernrail','brexit','ukip'],\
    ['rain','train','work','morning']]
    answer = []
    for term in terms:
        answer.append([term, model.doesnt_match(term)])
    print (tabulate(answer, headers=headers, tablefmt=STYLETABLE))

    terms = ['brexit', 'flood', 'weather', 'rain']
    mostlike = []
    headers = ['Keyword', 'The terms most similar to keyword']
    for term in terms:
        mostlike.append([term, model.most_similar(term)])
    print (tabulate(mostlike, headers=headers, tablefmt=STYLETABLE))

    comparisons = [('rain', 'flooding'), ('london','flooding'), ('essex','flooding'), ('floods','storms'), \
     ('thunderstorm', 'lightning'), ('weather', 'delays'), ('weather','trains'), ('boris','cameron'),\
      ('remain','brexit'), ('leave','brexit'), ('flood', 'brexit'),('voteleave', 'voteremain'), ('eu','brexit')]
    similarity = []
    headers = ['Similarity Between', 'Value']
    for item in comparisons:
        similarity.append( [','.join(item), str(model.similarity(item[0], item[1])) ])
    print (tabulate(similarity, headers=headers, tablefmt=STYLETABLE))

    
