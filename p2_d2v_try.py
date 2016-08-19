'''
PYTHON 2.* version as could not get tsne to work in v3 :( 
try out a gensim doc2vec model train on tweets 
'''
import struct;print struct.calcsize("P") * 8 # python
# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec, Word2Vec

# numpy
import numpy
import pandas as pd
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
# import jlpb
# import prepare

# get training and test data from mongodb:
if CURR_PLATFORM != 'linux':
    print('edit to add mongodb connexction')


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
        print 'for ' + count + ', you said ', response.upper()

        # TODO: update db record
    




def make_tsne(model):
    
    import time
    from sklearn.manifold import TSNE
    from sklearn.cluster import AffinityPropagation
    from sklearn.decomposition import PCA
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    n_sne = 7000

    time_start = time.time()
    
    
    dims_2 = []
    labels = list(model.docvecs.doctags.keys())

    

    # trial code: 
    # https://github.com/innainu/blog/blob/master/gizmodo_doc2vec/notebooks/doc2vec_gizmodo.ipynb
    # tsne_model = TSNE(n_components=2, random_state=0, verbose = 1, init= "pca")
    # tsne_articles_2D = tsne_model.fit_transform(model.docvecs)

    # kmeans = KMeans(5, n_jobs=-1)
    # top_cluster_results = kmeans.fit_predict(tsne_articles_2D[first_idx])

    # plt.scatter([x[0] for x in tsne_articles_2D[first_idx]],\
    #  [x[1] for x in tsne_articles_2D[first_idx]], c=top_cluster_results)

    # exit()
    # end trial code ====================



    #svd = TruncatedSVD(n_components=16, random_state=0)
    # , perplexity=10.0, learning_rate=200, n_iter=300, metric='cityblock'

    #model_svd = svd.fit_transform(model.docvecs)
    #print 'svdshape', model_svd.shape # 50k , 40

    tsnem = TSNE(n_components=2, \
         random_state=0, verbose=1, metric='euclidean')

    # tsne_ready = tsnem.fit_transform(model_svd)
    dims2 = []
    # print tsne_ready.shape
    # print dir(tsne_ready)
    
    for idx, doc in enumerate(model.docvecs):
        if idx % 10000 == 0:
            print 'doc: ', idx
        print doc
        print numpy.isfinite(doc).all() # True 
        print numpy.isnan(doc).all() # False
        print numpy.isinf(doc).all() # False
        print 'done'
        
        dims_2.append(TSNE().fit_transform(doc).tolist()[0])

    print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)

    # cache tsne as CSV
    numpy.savetxt('doc2vec_tsne.csv', dims_2, delimiter='\t')

    print 'clusters:'

    affp = AffinityPropagation().fit(dims_2)

    print 'Preparing plot...'
    plot_cluster(affp, dims_2, labels)

    # return np.asarray(dims_2) 

def plot_cluster(af, doc_2d, fnames):
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)

    #print(cluster_centers_indices)
    #print(len(af.labels_))
    #print(len(labels))
    #print(n_clusters_)

    plt.figure(num=1, figsize=(80, 80), facecolor="w", edgecolor="k")
    colors = cycle("bgrcmyk")

    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k         # class_members ist array von boolschen werten, beschreibt cluster membership
        cluster_center = doc_2d[cluster_centers_indices[k]]

        fnames_cluster = []
        fname_indices = [i for i, x in enumerate(class_members) if x]
        for i in fname_indices: fnames_cluster.append(fnames[i])

        #print(fnames_cluster)
        #print(len(class_members))
        #print(len(fnames))
        #print(cluster_center)

        plt.plot(doc_2d[class_members, 0], doc_2d[class_members, 1], col + ".")
        plt.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor=col, markersize=20)

        #plt.annotate(fnames[labels[k]], (cluster_center[0], cluster_center[1]), xytext=(0, -8),
        #        textcoords="offset points", va="center", ha="left")

        for x, fname in zip(doc_2d[class_members], fnames_cluster):
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col, linestyle='--', linewidth=1)
            plt.annotate(fname, (x[0], x[1]), xytext=(0, -8),
                        textcoords="offset points", va="center", ha="left")

    plt.savefig("out_doc2vec.png", facecolor="w", dpi=90)
    print "saved output to ./out_doc2vec.png\n"



if __name__ == '__main__':   

    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
     level=logging.INFO)

    # What are we doing with this run of the script? 
    # 
    WRITE_OUT = False
    CREATE_MODEL = False # set false to load the model with chosen parameters

    fout = 'model_d2v.d2v' #output filename suffix

    # train and test sets set up:
    files = ['train_pos_frm500.txt','train_neg_frm500.txt',\
     'test_pos_frm500.txt', 'test_neg_frm500.txt', 'unlabelled.txt']
    
    ftrainpos = open(files[0], 'a') 
    ftrainneg = open(files[1], 'a')
    ftestpos = open(files[2], 'a') 
    ftestneg = open(files[3], 'a')
    funlabelled = open(files[4], 'a')

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

    #####
    #
    # PARAMS for Word2Vec & Doc2Vec =======================================
    #
    #
    epochs = 16
    vec_length = 32 # rule thumb, sqrt of vocab.
    window=8
    vec_type = 'doc2vec'
    negsample=6
    sample=1e-5
    #
    # END PARAMS for Word2Vec & Doc2Vec =======================================
    #
    #####
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
        # model = Doc2Vec.load('./sample_neg' + str(negsample) + '_' + str(epochs) + '_' + str(vec_length) + fout)
        model = Doc2Vec.load('./sample_neg6_16_32model_d2v.d2v')
        tsne = make_tsne(model)

        doc_orig = model.syn0
        # numpy.savetxt('doc2vec_model.csv', doc_orig, delimiter='\t')
        # --  a dict (like `vocab` in Word2Vec) ..
        # where the keys are all known string tokens (aka 'doctags')
        labels = list(model.docvecs.doctags.keys())
        # build list of tags from the metadata
        df = pd.DataFrame(index=labels, columns=['Words'])
        print df.describe()

        # a list of all sentence labels -- long!!
        # print(model.docvecs.offset2doctag) #   e.g. TEST_POS_0 , TRAIN_UNS_0  etc
        
        # NB can refer to sepcific doc vector then, like so:
        dockey ='TEST_POS_0'
        docvec = model.docvecs[dockey] 
        print dockey, docvec
        # TODO: could try compare some flood - tweets here and then also with non flood tweets 
        # for distance assessment

        print df.iloc[95:100, :10] # alt way of doing head()
        print '\n\n'
        print df.tail() 

    else:
        model = Word2Vec.load('./' + str(epochs) + '_' + str(vec_length) + fout)

    



    
