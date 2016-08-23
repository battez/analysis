'''
try out a gensim doc2vec model train on tweets 
'''

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
    docs = dbt.find()[3023:]
    count = 0
    response = ''

    for doc in docs:

        place = doc['place']['full_name']
        
        stop = False
        count += 1

        while True:
            response = input( jlpb.uprint(doc['id_str'] + ': ' + doc['text'] + \
                        "\n user:" + doc['user']['screen_name'] + \
                            "\n place:" + place) )
            if response == 'x':
                stop = True
                break
            elif response == 'y':
                # update db record
                dbt.update_one({
                  '_id': doc['_id']
                },{
                  '$set': {
                    't_class': 1
                  }
                }, upsert=False)
                response = 'yes'
                
            else:
                response = 'n'

            print('...for ' + str(count) + ', you said ', response.upper())
            print('------------------------------------')
            break

        if stop:
            break

    print('docs complete: ', count)


def make_tsne(model):
    '''
    do an SVD of the model of Doc2Vec, then continue to tSNE with 2d projection.
    '''
    import numpy as np
    from sklearn.manifold import TSNE
    from sklearn.cluster import AffinityPropagation
    from sklearn.decomposition import PCA
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    labels = list(model.docvecs.doctags.keys())
    print(model.docvecs.most_similar(20))

    z = list(model.docvecs)

    print(len(z))
    z = z[:15000]
    print('shape')
    
    # we reduce to most important components to a number tSNE can then handle:
    svd = TruncatedSVD(n_components=50, random_state=0, n_iter=10)    
    model_svd = svd.fit_transform(model.docvecs)
    print('svdshape', model_svd.shape) # 50k , num vector dims in params of d2v

    import uuid
    unique_suffix = str(uuid.uuid4())
    # save the SVD
    np.savetxt('doc2vec_tsne_svd' + unique_suffix + '.csv', model_svd, delimiter='\t')

    # Can use simple plot examples: 
    # http://alexanderfabisch.github.io/t-sne-in-scikit-learn.html
    # but passing to Bokeh for now.  
    # this will iterate until no more progress is made or use n_iter= param
    # euclidean is squared euclid distance
    tsnem = TSNE(n_components=2, random_state=0, verbose=1, metric='euclidean',\
     learning_rate=100, n_iter=400)

    # NB angle is tradeoff accuracy vs process time, higher is more accurate.
    trimmed = model_svd[:15000]
    trimmed_nosvd = z
    tsne_ready = tsnem.fit_transform(trimmed_nosvd)

    print(tsne_ready.shape)
    try:
        print(tsne_ready[0])
    except:
        pass

    # save the tSNE
    np.savetxt('doc2vec_tsne' + unique_suffix + ' .csv', tsne_ready, delimiter='\t')

    plot_tsne(False, tsne_ready)



def plot_tsne(file=False, data=False):
    '''
    use Bokeh lib to plot the tSNE projection into 2D data viz
    '''
    x, y = ([], [])

    if file:
        import csv
        with open(file) as csv_file:
            read_csv = csv.reader(csv_file, delimiter='\t')
           
            for idx, row in enumerate(read_csv):
                x.append(row[0])
                y.append(row[1])
                if idx > 50000:
                    break
    else:

        x = data[:, 0]
        y = data[:, 1]


    import bokeh.plotting as bp
    from bokeh.models import HoverTool, BoxSelectTool
    from bokeh.plotting import figure, show, output_file
    import pandas as pd
    
    output_file('svd_tsne.html',title='TSNE tweets data')
    plot_d2v = bp.figure(plot_width=1200, plot_height=900, title="tSNE tweets (doc2vec)",
        tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
        x_axis_type=None, y_axis_type=None, min_border=1)

    # make a pandas dataframe for x and y and the original text:
    dfcsv = join_data()
    print(dfcsv.shape)
    df = pd.DataFrame({'x':x, 'y': y, 'tweet':dfcsv.tweet[0:15000]})
    print(df.shape)
    jlpb.uprint(df.describe())

    # source = ColumnDataSource({'x': df.x, 'y': df.y, 'tweet': df.tweet})
    source = bp.ColumnDataSource(data=df)
    plot_d2v.scatter(x=x, y=y, source=source )

    hover = plot_d2v.select(dict(type=HoverTool))
    hover.tooltips = [("xval", "@x"),  ("yval", "@y"), ("text", "@tweet"), ('index', '$index')]

    hover.mode = 'mouse'

    show(plot_d2v)
    print('plotted; done.')


def join_data(files=['train_pos_frm500.txt','train_neg_frm500.txt',\
     'test_pos_frm500.txt', 'test_neg_frm500.txt', 'unlabelled.txt']):
    '''
    Join our training set and unlabelled set into one dataframe 
    with one column
    '''
    import pandas as pd  
    all_rows = None
    for dummy, txt in enumerate(files):
        df = pd.read_csv(txt, header=None,  sep=',')
        df.rename(columns={ 0 :'tweet'}, inplace=True)
        if isinstance(all_rows, pd.DataFrame):
            all_rows = pd.concat([all_rows, df], axis=0)
        else:
            all_rows = df
    return all_rows 


def tokenise_csv(file='buffer10k4002set.csv'):
    '''
    convert a csv of tweets into tokenised csv saved with new filename.
    '''
    import csv
   
    # open an output file to save to:
    prefix = 'tokens_'
    output = open(prefix + file, 'a', encoding='utf8')

    # open a CSV and read all its lines
    with open(file, encoding='utf-8') as csv_file:

        docs = csv.reader(csv_file, delimiter=',')
        for idx, doc in enumerate(docs):
        
            # normalise tweet text
            text = prepare.normalise_tweet(doc[1], unicode_replace=False)
            
            # write to file with label
            output.write(doc[0] + ',' + text + '\n')
           


if __name__ == '__main__':   

    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
     level=logging.INFO)

    ##
    ## VARIOUS maintenance tasks:
    #uncomment to begin labelling tweets:
    # quick_label_tweets()

    # d2file = 'doc2vec_tsne_svd.csv'
    # plot_tsne(d2file) # add 1 to the index to get the line number..
    ## END VARIOUS mainitenace tasks
    ##

    # IMPORTANT: Set the mode here for this run of the script:
    WRITE_OUT = False
    CREATE_MODEL = False # set false to load the model with chosen parameters
    load_only = 'default_sample_neg5_16_104model_d2v.d2v' # set to the model you wish to load

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

    #####
    #
    # PARAMS for Word2Vec & Doc2Vec =======================================
    #
    #
    epochs = 16
    vec_length = 104 # rule thumb, sqrt of vocab.
    window = 8
    vec_type = 'doc2vec'
    negative = 5
    sample = 1e-4
    #
    # END PARAMS for Word2Vec & Doc2Vec =======================================
    #
    #####
    if CREATE_MODEL:

        # sources with tweets in for unsupervised trainging of d2v model.
        sources = { files[4]:'TRAIN_UNS'} 
        # using just unlabelled for now 
        '''files[0]:'TRAIN_POS',\
        files[1]:'TRAIN_NEG',\
        files[2]:'TEST_POS',\
        files[3]:'TEST_NEG',\
        '''


        # FIXME: should convert LabeleledLineSentence to Tagged Document
        # since it is deprecated.
        # 
        sentences = LabeledLineSentence(sources)

        # dynamically get the workers for this machine:
        import multiprocessing
        cores = multiprocessing.cpu_count()


        if vec_type == 'doc2vec':
            #  
            # model = Doc2Vec(min_count=1, window=window, negative=negative, size=vec_length,\
            #  workers=cores, sample=sample, dm=1, dm_concat=1)
            model = Doc2Vec(min_count=1,size=vec_length,\
             workers=cores)
            model.build_vocab(sentences.to_array())

            # training according to number of Epochs required:
            for epoch in range(epochs):
                # this shuffles the data around to make this balanced in terms of the 
                # learning rate of the procedure. 
                model.train(sentences.sentences_perm())

        else:
            model = Word2Vec(min_count=10, window=window, size=vec_length, workers=cores)
            model.build_vocab(sentences)
            # more is better but longer... ~ 20 ideal
            for epoch in range(epochs):
                '''
                in every training epoch, the set of sentences input to the model 
                gets randomised
                ''' 
                model.train(sentences.sentences_perm())

        # Save the model to disk, including its params:
        most_recent = './default_sample_neg' + str(negative) + '_' + \
            str(epochs) + '_' + str(vec_length) + fout
        model.save(most_recent)


    # Load model stage:
    if vec_type == 'doc2vec':
        '''
        Load in the model and pass to tSNE
        '''
        try:
            most_recent
        except:
            most_recent = load_only 
       
        print('model used:', most_recent)
        model = Doc2Vec.load(most_recent)

        # Dimensionality Reduction using tSNE
        tsne = make_tsne(model)

        # doc_orig = model.syn0
        # numpy.savetxt('doc2vec_model.csv', doc_orig, delimiter='\t')
        # --  a dict (like `vocab` in Word2Vec) ..
        # where the keys are all known string tokens (aka 'doctags')
        labels = list(model.docvecs.doctags.keys())
        # build list of tags from the metadata
        df = pd.DataFrame(index=labels, columns=['Words'])
        print(df.describe())

        # a list of all sentence labels -- long!!
        # print(model.docvecs.offset2doctag) #   e.g. TEST_POS_0 , TRAIN_UNS_0  etc
        
        # NB can refer to sepcific doc vector then, like so:
        dockey ='TEST_POS_0'
        docvec = model.docvecs[dockey] 
        print(dockey, docvec)
        # TODO: could try compare some flood - tweets here and then also with non flood tweets 
        # for distance assessment

        print(df.iloc[95:100, :10]) # alt way of doing head()
        print('\n\n')
        print(df.tail())

    else:
        # TODO: word2Vec method not implemented yet.
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

    
