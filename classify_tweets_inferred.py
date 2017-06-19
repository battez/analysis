'''
local machine test, autumn 2016. 
'''
import pandas as pd
import random
import numpy as np
import os, sys

# specify dir to save a pickle model here, later
dir_model = os.path.expanduser('~/Downloads/')
CURR_PLATFORM = sys.platform

# Mac vs Win vs Linux
if CURR_PLATFORM == 'darwin':
    TWITDIR = os.path.expanduser('~/Dropbox/data-notes-mac-to-chrome/data-incubator/Project_submission/supporting_files_code_queries_logs_Etc/demoapptwitter')
    SCRAPEDIR = os.path.expanduser('~/Dropbox/data-notes-mac-to-chrome/data-incubator/Project_submission/supporting_files_code_queries_logs_Etc/scrape')
elif CURR_PLATFORM != 'linux':
    TWITDIR = 'U:\Documents\Project\demoapptwitter'
    SCRAPEDIR = 'U:\Documents\Project\scrape'
    dir_model = 'C:/Users/johnbarker/Downloads/'
else:
    TWITDIR = '/home/luke/programming/'
    SCRAPEDIR = '/home/luke/programming/scraping'

sys.path.insert(0, TWITDIR)
sys.path.insert(0, SCRAPEDIR)

# get some handy functions 
import jlpb
import jlpb_classify


if __name__ == "__main__":
    '''
    We will use a Doc2Vec model to then train a classifier
    (Logistic Regression currently) to try to classify relevant
    tweets from irrelevant ones for topic of flooding.

    '''
    import time
    start = time.perf_counter() # keep track of processing time

    # Set up some key variables for the process of model training:
    # Use a random seed for reproducibility 
    seed = 40 
    # This can take a LOT of time if high! but should give better
    # performance for the classifier. 
    epochs = 16 
    vocab_rows = 50000 # how many unlabelled tweets to use for building vocab in D2Vec
    vocab_frac = 1 # when using a sample of a huge file of unlabelled tweets
    vecs = 160
    test_num = 450 # 450

    ## LOAD DATA SETS OF TWEETS TO PANDAS DFs FROM CSV==========================
    ##
    # Make Pandas dataframes from labelled data and combine to 
    # a single dataframe that we then split into test and training set:
    #
    df = pd.read_csv('pos_frm500.csv')
    df = df[[u'label',u'text']]

    ndf = pd.read_csv('neg_frm500.csv')
    ndf = ndf[[u'label',u'text']]

    ## combine these two together: 
    zdf = pd.concat([df, ndf], axis=0)


    # load this further 4002 labelled rows :
    xdf = pd.read_csv('tokens_buffer10k4002set.csv')
    xdf = xdf[[u'label',u'text']]

    # Combine these 500 + 4002 labelled:
    tdf = pd.concat([zdf, xdf], axis=0)

    # make the string a list 
    # NB needed in D2V gensim: (ie split the string into words, for D2V)
    tdf.loc[:,'text'] = tdf.loc[:,'text'].map(jlpb_classify.split)
    
    print('tdf',tdf.shape)

    tdf[tdf == 'negative'] = 0
    tdf[tdf == 'positive'] = 1
    print('head:5', tdf.head(5))
    print('tail:5', tdf.tail(5))
    # Randomise the order of the Labelled set rows 
    # (using a seed for reproducibility)
    tdf = tdf.sample(frac=1, random_state=np.random.RandomState(seed)).\
        reset_index(drop=True)

    print('Head labelled set', tdf.head(5))
    print('Tail labelled set', tdf.tail(5))
    print('Dims tdf',tdf.shape)


    # Load in our unlabelled data set of tweets to build the d2v vocabulary.
    print('loading unlabelled vocab tweets into dataframe...')
    udf = pd.read_csv('unlab420k.csv') #unlabelled.csv has 50k
    udf = udf[[u'text']]
    print('completed loading unlabelled.')
    
    # udf = udf.sample(frac=vocab_frac, random_state=np.random.RandomState(seed)).\
    #     reset_index(drop=True) 
    # udf = udf.iloc[:5] # debug
    # uncomment to use with 23rd data only:
    # udf = udf.iloc[:vocab_rows]
    print(udf.size , 'rows')


    # we need to clean up chars in this unlabelled tweets and tokenise into words: 
    udf.loc[:,'text'] = udf.loc[:,'text'].map(jlpb_classify.split)
    total_num_unlabelled = udf.size

    # Gensim Doc2Vec for high-dim vectors in model(s) for each tweet:
    from gensim.models.doc2vec import TaggedDocument, Doc2Vec

    total_num = int(tdf.size/2)
    print('tweets data dims: ', total_num)
    
    

    # split for the needed test and training data 
    # maintain approx. a 9:1 ratio of training:test,
    # as we have relatively little labelled data.
    
    print('Test set size', test_num)
    training_num = total_num - test_num
    print('Training set size', training_num)


    documents = [TaggedDocument(list(tdf.loc[i,'text']),[i]) for i in range(0, total_num)]
    documents_unlabelled = [TaggedDocument(list(udf.loc[i,'text']), \
        [i]) for i in range(0, total_num_unlabelled)]

    documents_all = documents_unlabelled

    doc2vec_train_id = list(range(0, total_num_unlabelled))
    #random.shuffle(doc2vec_train_id)

    # training documents for Doc2Vec 
    training_doc = [documents_all[id] for id in doc2vec_train_id]
    print('num training docs: ', len(training_doc))
    # get all class labels 
    class_labels = tdf.loc[:,'label']

    # find out max system workers from the cores
    # so we can make use of the max CPU:
    import multiprocessing
    cores = multiprocessing.cpu_count()

    # build fresh doc2vec models if True set below!
    # (otherwise load from disk)
    most_recent = dir_model + 'Mac_d2v_tol0001_win10_420kseed40_160se4_ep16_minc3'
    # save current labelled dataframe to a CSV 
    tdf.to_csv(most_recent + 'LOG.csv')

    model_DM, model_DBOW = (None, None)

    # change below line to True to load in new models:
    if False:
        # Parameters can be adjusted to try to get better accuracy from classifier.
        model_DM = Doc2Vec(size=vecs, window=10, min_count=3, sample=1e-4,\
         negative=5, workers=cores,  dm=1, dm_concat=1 )
        model_DBOW = Doc2Vec(size=vecs, window=10, min_count=3, sample=1e-4,\
         negative=5, workers=cores, dm=0)

        # construct the vocabs for our models
        model_DM.build_vocab(training_doc)
        model_DBOW.build_vocab(training_doc)

        for it in range(0,epochs):
            # progress as this takes a long time:
            if (it % 2) == 0:
                print('epoch ' + str(it) + ' of ' + str(epochs))

            random.shuffle(doc2vec_train_id)
            training_doc = [documents_all[id] for id in doc2vec_train_id]
            
            # Train the model again
            model_DM.train(training_doc)
            model_DBOW.train(training_doc)

        # Save the trained models:
        fout = 'DM.d2v'
        model_DM.save(most_recent + fout)
        model_DM.init_sims(replace=True)

        fout = 'DBOW.d2v'
        model_DBOW.init_sims(replace=True)
        model_DBOW.save(most_recent + fout)

    else:
        # Load Doc2Vec model from disk:
        fout = 'DM.d2v'
        model_DM = Doc2Vec.load(most_recent + fout)

        fout = 'DBOW.d2v'
        model_DBOW = Doc2Vec.load(most_recent + fout)


    # train the two different methods of the Doc2Vec algorithm:
    # NB DBOW is more similar to the recommended skip-gram of 
    # Word2Vec by the original paper's authors.  
 

    print('nonmatch', model_DM.doesnt_match("delay government flooding lightning".split()))

    print('nonmatch', model_DM.doesnt_match("euref voteout remain lightning".split()))
    print('euref sim by word', model_DM.similar_by_word('euref'))
    print('flood ', model_DM.similar_by_word('flood'))
    print('flooding ', model_DM.similar_by_word('flooding'))

    print('weather', model_DM.most_similar('weather'))
    print('rain', model_DM.most_similar('rain'))
    print('lightning', model_DM.most_similar('lightning'))
    print('thunder', model_DM.most_similar('thunder'))
    print('thunderstorm', model_DM.most_similar('thunderstorm'))
    print('ukstorm', model_DM.most_similar('ukstorm'))

    print('trains', model_DM.most_similar('trains'))
    print('delays', model_DM.most_similar('delays'))

    print('ligh thun similarity', model_DM.similarity('lightning', 'thunder'))
    
    '''
    Use Logistic Regression and train a classifier from Doc2Vec model and labelled data.

    Then output plots of confusion matrices of the accuracy of the model applied to 
    test data.

    Credit: https://www.zybuluo.com/HaomingJiang/note/462804
    NB Adapted methodology from a tutorial in Doc2Vec -- URL above -- to scaffold 
    this classification.

    '''
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    import statsmodels.api as sm


    random.seed(100) # 100 , 1212
    new_index = random.sample(range(0, total_num), total_num)

    # set the IDs for the test set:
    testID = new_index[-test_num:]

    # set the IDs for the training set
    trainID = new_index[:-test_num]

    train_targets, train_regressors = zip(*[(class_labels[id], \
        list(model_DM.docvecs[id]) + list(model_DBOW.docvecs[id])) for id in trainID])

    # add a constant term so that we fit the intercept of our linear model.
    #  i.e. the log odds *only* when x1=x2=0, desirable to avoid biasing the model.
    train_regressors = sm.add_constant(train_regressors)

    # set params for model: tol=0.0001,
    model_logreg = LogisticRegression(C=0.5, tol=0.0001, penalty='l2',  n_jobs=-1)

    model_logreg.fit(train_regressors, train_targets) 

    ## Prepare the test data for testing the model:
    accuracies = []
    test_regressors = [list(model_DM.docvecs[id]) + list(model_DBOW.docvecs[id]) for id in testID]

    # add a constant term so that we fit the intercept of our linear model.
    test_regressors = sm.add_constant(test_regressors)
    test_predictions = model_logreg.predict(test_regressors)
    accuracy = 0

    # Loop through the test predictions and adjust accuracy measurement
    # Also print out the correct positive and all incorrect predictions.
    for i in range(0, test_num):
        if test_predictions[i] == tdf.loc[testID[i],u'label']:
            
            if(test_predictions[i] == 1):
                jlpb.uprint('Correct: id', str(i) + ', tdf_row:' + str(testID[i]) + ', ', \
                    str(tdf.loc[testID[i], u'label']),\
                    tdf.loc[testID[i], u'text'])
            accuracy = accuracy + 1
        else:
            jlpb.uprint('WRONG:'+ str(i) + ', tdf_row:' + str(testID[i]) +', should be', \
                str(tdf.loc[testID[i], u'label']), tdf.loc[testID[i], u'text'])

    # calculate the final accuracy:        
    accuracies = accuracies + [1.0 * accuracy / test_num]

    ## Show user time needed for this classifier:
    total = int((time.perf_counter() - start) / 60)
    print("Process took %s minutes" % total)

    ###
    # Infer vectors checks of the model:
    ###
    doc_id = np.random.randint(model_DM.docvecs.count)  # pick random doc; re-run cell for more examples
    print('for doc %d...' % doc_id)
    print(udf.loc[doc_id, 'text'])
    inferred_docvec = model_DM.infer_vector(udf.loc[doc_id, 'text'])
    inferdocsim = model_DM.docvecs.most_similar([inferred_docvec], topn=5)
    print('DM:\n %s' % (inferdocsim))
    for doc in inferdocsim:
        print('doc: ')
        
        print(udf.loc[doc[0], 'text'])
        

    extra_docs = 'flood warning bishops waltham and botley on the river hamble june'.split() 
    extravec = [model_DM.infer_vector(extra_docs)]
    extramostsim = model_DM.docvecs.most_similar(extravec)
    print (extramostsim)

    for doc in extramostsim:
        print('doc: ', doc[0])
        
        print(udf.loc[doc[0], 'text'])
        
    extra_docs = 'crazy weather pound lane onto a is flooding water is expelling from the drainage a has excessive water'.split() 
    extravec = [model_DM.infer_vector(extra_docs)]
    extramostsim = model_DM.docvecs.most_similar(extravec)
    print (extramostsim)

    for doc in extramostsim:
        print('doc: ', doc[0])
        
        print(udf.loc[doc[0], 'text'])


    exit('aborting...')

    extra = [ list(model_DM.infer_vector(extra_docs)) + list(model_DBOW.infer_vector(extra_docs)) ]


    print(extra)




    # add a constant term so that we fit the intercept of our linear model.
    extra_regressors = sm.add_constant(extra_regressors)
    print (model_logreg.predict(extra))


    ## OUTPUT Evaluation of Accuracy=================================================
    # Accuracy rates and so on:
    #
    # Produce some confusion matrices and plot them:
    #
    # cast the labels for the confusion matrix, otherwise they are seen as binary!
    cast = tdf.loc[testID, u'label']
    cast = (cast.values).astype(np.int8) # numpy.ndarray now!

    confusion_mtx = confusion_matrix(test_predictions, cast)
    print('test conf matrix: ', confusion_mtx)
    jlpb_classify.show_confusion_matrix(confusion_mtx)

    train_predictions = model_logreg.predict(train_regressors)
    accuracy = 0
    for i in range(0,len(train_targets)):
        if train_predictions[i] == train_targets[i]:
            accuracy = accuracy + 1
    accuracies = accuracies + [1.0 * accuracy / len(train_targets)]
    confusion_mtx = confusion_matrix(train_predictions,train_targets)
    print('training conf matrix: ', confusion_mtx)
    jlpb_classify.show_confusion_matrix(confusion_mtx)


    ## Todo: Further evaluations
    ## Show F1 score
    '''
    Because precision and recall both provide valuable 
    information about the quality of a classifier, you often 
    want to combine them into a single general-purpose score. 
    The F1 score is defined as the harmonic mean of recall and precision:

    F1 = (2 x recall x precision) / (recall + precision)

    The F1 score thus tends to favor classifiers that are strong in 
    both precision and recall, rather than classifiers that 
    emphasize one at the cost of the other.
    '''

    ## Todo:
    ## predict unseen and unlabelled tweets
    ##
    ##
    ## make token lists first
    #tokens = "ACTUAL SENTENCE".split()  # should be same tokenization as training
    # then make document vectors 
    #dv = model.infer_vector(tokens)     # note: may want to use many more steps than default
    # try the similar tweets:
    #sims = model.docvecs.most_similar(positive=[dv])
   
    # then model_logreg.predict() / 
    # model_logreg.predict_proba() to get the class probabailities


    

    
