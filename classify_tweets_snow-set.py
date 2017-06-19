'''
local machine test, autumn 2016. 
'''
import pandas as pd
import random
import numpy as np
import os, sys

# specify dir to save a pickle model here, later
subdir = 'snowset/'
dir_model = os.path.expanduser('~/Downloads/') + subdir

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
    seed = 7777 #40 
    # This can take a LOT of time if high! but should give better
    # performance for the classifier. 
    epochs = 30 
    vocab_rows = 11717 # how many unlabelled tweets to use for building vocab in D2Vec
    vocab_frac = 1 # when using a sample of a huge file of unlabelled tweets
    vecs = 200
    test_num = 250 # total labelled minus this will be the training set!

    # doc2vec params:
    min_count = 4
    window = 10
    sample = 1e-5
    negative = 5
    print ('Parameters: seed',seed,'epochs',epochs,'vocab_rows',vocab_rows,'vecs',vecs,'test_num',test_num,\
        'min_count',min_count,'window',window,'sample',sample,'negative',negative)
    # find out max system workers from the cores
    # so we can make use of the max CPU:
    import multiprocessing
    workers = multiprocessing.cpu_count()

    ## LOAD DATA SETS OF TWEETS TO PANDAS DFs FROM CSV==========================
    ##
    # Make Pandas dataframes from labelled data and combine to 
    # a single dataframe that we then split into test and training set:
    #
    subdir = 'snowset/'
    df = pd.read_csv(subdir + 'has_label.csv')
    tdf = df[[u'label',u'text']]

    # make the string a list 
    # NB needed in D2V gensim: (ie split the string into words, for D2V)
    tdf.loc[:,'text'] = tdf.loc[:,'text'].map(jlpb_classify.split)
    
    print('tdf',tdf.shape) # num rows labelled data

    tdf[tdf == 'negative'] = 0
    tdf[tdf == 'positive'] = 1
    # jlpb.uprint('head:10', tdf.head(10))
    # jlpb.uprint('tail:10', tdf.tail(10))
    
    # Randomise the order of the Labelled set rows 
    # (using a seed for reproducibility)
    tdf = tdf.sample(frac=1, random_state=np.random.RandomState(seed)).\
        reset_index(drop=True)

    # jlpb.uprint('Head labelled set', tdf.head(10))
    # jlpb.uprint('Tail labelled set', tdf.tail(10))
    

    # Load in our unlabelled data set of tweets to build the d2v vocabulary.
    print('loading unlabelled vocab tweets into dataframe...')
    udf = pd.read_csv(subdir+'unlabelled11k.csv') 
    udf = udf[[u'text']]
    print('completed loading!')

    print('randomising unlabelled dataframe...')
    # 100k+ tweets
    udf = udf.sample(frac=vocab_frac, random_state=np.random.RandomState(seed)).\
        reset_index(drop=True) 
    # udf = udf.iloc[:5] # debug
    # uncomment to use with 23rd data only:
    # udf = udf.iloc[:vocab_rows]
    print(udf.size , 'rows')
    print('completed randomising!')

    # we need to clean up chars in this unlabelled tweets and tokenise into words: 
    udf.loc[:,'text'] = udf.loc[:,'text'].map(jlpb_classify.split)
    total_num_unlabelled = udf.size

    # Gensim Doc2Vec for high-dim vectors in model(s) for each tweet:
    from gensim.models.doc2vec import TaggedDocument, Doc2Vec

    total_num = int(tdf.size/2)
    print('tweets data dims: ', total_num)
    

    # Split for the needed test and training data 
    # maintain approx. a 9:1 ratio of training:test,
    # as we have relatively little labelled data.
    print('Test set size', test_num)
    training_num = total_num - test_num
    print('Training set size', training_num)


    documents = [TaggedDocument(list(tdf.loc[i,'text']),[i]) for i in range(0, total_num)]
    documents_unlabelled = [TaggedDocument(list(udf.loc[i,'text']), \
        [i+total_num]) for i in range(0, total_num_unlabelled)]
    documents_all = documents + documents_unlabelled

    doc2vec_train_id = list(range(0, total_num + total_num_unlabelled))
    random.shuffle(doc2vec_train_id)

    # training documents for Doc2Vec 
    training_doc = [documents_all[id] for id in doc2vec_train_id]

    # get all class labels 
    class_labels = tdf.loc[:,'label']

    

    # build fresh doc2vec models if True set below!
    # (otherwise load from disk)
    most_recent = dir_model  + 'Mac_d2v_tol0001_win10_11kseed7777_200se4_ep30_minc4'
    # save current labelled dataframe to a CSV 
    tdf.to_csv(most_recent + 'LOG.csv')

    model_DM, model_DBOW = (None, None)

    # change below line to True to load in new models:
    if False:
        # Parameters can be adjusted to try to get better accuracy from classifier.
        model_DM = Doc2Vec(size=vecs, window=window, min_count=min_count, sample=sample,\
         negative=negative, workers=workers,  dm=1, dm_concat=1 )
        model_DBOW = Doc2Vec(size=vecs, window=window, min_count=min_count, sample=sample,\
         negative=negative, workers=workers, dm=0)

        # construct the vocabs for our models
        model_DM.build_vocab(training_doc)
        model_DBOW.build_vocab(training_doc)

        for it in range(0, epochs):
            # progress as this takes a long time:
            if (it % 2) == 0:
                print('epoch ' + str(it) + ' of ' + str(epochs))

            random.shuffle(doc2vec_train_id)
            training_doc = [documents_all[id] for id in doc2vec_train_id]
            model_DM.train(training_doc)
            model_DBOW.train(training_doc)

        fout = 'DM.d2v'
        # model_DM.init_sims(replace=True)
        model_DM.save(most_recent + fout)

        fout = 'DBOW.d2v'
        # model_DBOW.init_sims(replace=True)
        model_DBOW.save(most_recent + fout)

    else:
        # Load Doc2Vec model from disk:
        fout = 'DM.d2v'
        model_DM = Doc2Vec.load(most_recent + fout)
        print ('shape synset DM:', model_DM.syn0.shape)
        fout = 'DBOW.d2v'
        model_DBOW = Doc2Vec.load(most_recent + fout)


    # train the two different methods of the Doc2Vec algorithm:
    # NB DBOW is more similar to the recommended skip-gram of 
    # Word2Vec by the original paper's authors.  
 
    print('snow ', model_DM.similar_by_word('snow'))
    print('day', model_DM.most_similar('day'))
    print('snowday', model_DM.most_similar('snowday'))

    print('cars', model_DM.most_similar('cars'))
    print('roads', model_DM.most_similar('roads'))
    print('road', model_DM.most_similar('road'))
    print('ditch', model_DM.most_similar('ditch'))
    print('safe', model_DM.most_similar('safe'))
    print('plow', model_DM.most_similar('plow'))

    # print('towed', model_DM.most_similar('towed'))

    print('weather', model_DM.most_similar('weather'))
    print('school', model_DM.most_similar('school'))
    print('iowa', model_DM.most_similar('iowa'))
    print('(State abbrev.) mn', model_DM.most_similar('mn'))

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
    import matplotlib.pyplot as plt




    '''
    Dim. reduction to visualise our doc vectors:
    
    from sklearn.manifold import TSNE
    from sklearn.decomposition import TruncatedSVD
    
    vectors = list(model_DM.docvecs)[:4000]
    print(len(vectors))
    X_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(vectors)
    print(X_reduced.shape)
    X_embedded = TSNE(n_components=2, perplexity=40, verbose=1,\
        metric='euclidean', learning_rate=100, n_iter=200).fit_transform(X_reduced)
    print(X_embedded.shape)
    # plot the result
    vis_x = X_embedded[:, 0]
    vis_y = X_embedded[:, 1]

    plt.scatter(vis_x, vis_y)
    # , c=y_data, cmap=plt.cm.get_cmap("jet", 10)
    # plt.colorbar(ticks=range(10))
    # plt.clim(-0.5, 9.5)
    plt.show()
    exit()
    '''


    random.seed(50) # 100 , 1212
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
    model_logreg = LogisticRegression(C=3, tol=0.0001, penalty='l2',  n_jobs=-1)

    model_logreg.fit(train_regressors, train_targets) 

    ## Prepare the test data for testing the model:
    accuracies = []
    test_regressors = [list(model_DM.docvecs[id]) + list(model_DBOW.docvecs[id]) for id in testID]

    # add a constant term so that we fit the intercept of our linear model.
    test_regressors = sm.add_constant(test_regressors)
    print('shape testregressors',test_regressors.shape)
    test_predictions = model_logreg.predict(test_regressors)
    accuracy = 0

    # Loop through the test predictions and adjust accuracy measurement
    # Also print out the correct positive and all incorrect predictions.
    for i in range(0, test_num):
        print('probs for this prediction:', model_logreg.predict_proba(test_regressors[i]))
        
        if test_predictions[i] == tdf.loc[testID[i], u'label']:
            
            if test_predictions[i]:
                jlpb.uprint('True Pos: id', str(i) + ', tdf_row:' + str(testID[i]) + ', ', \
                    str(tdf.loc[testID[i], u'label']),\
                    tdf.loc[testID[i], u'text'])
            else:
                jlpb.uprint(tdf.loc[testID[i], u'text'])

            accuracy = accuracy + 1
        else:
            jlpb.uprint('FALSE:'+ str(i) + ', tdf_row:' + str(testID[i]) +', actual', \
                str(tdf.loc[testID[i], u'label']), tdf.loc[testID[i], u'text'])

    # calculate the final accuracy:        
    accuracies = accuracies + [1.0 * accuracy / test_num]

    ## Show user time needed for this classifier:
    total = int((time.perf_counter() - start) / 60)
    print("Process took %s minutes" % total)

    ## OUTPUT Evaluation of Accuracy=================================================
    # Accuracy rates and so on:
    #
    # Produce some confusion matrices and plot them:
    #
    # cast the labels for the confusion matrix, otherwise they are seen as binary!
    cast = tdf.loc[testID, u'label']
    cast = (cast.values).astype(np.int8) # numpy.ndarray now!

    confusion_mtx = confusion_matrix(cast, test_predictions)
    print('test conf matrix: ', confusion_mtx)
    jlpb_classify.show_confusion_matrix(confusion_mtx)

    train_predictions = model_logreg.predict(train_regressors)
    accuracy = 0
    for i in range(0, len(train_targets)):
        if train_predictions[i] == train_targets[i]:
            accuracy = accuracy + 1
    accuracies = accuracies + [1.0 * accuracy / len(train_targets)]
    confusion_mtx = confusion_matrix(train_targets, train_predictions)
    print('training conf matrix: ', confusion_mtx)
    jlpb_classify.show_confusion_matrix(confusion_mtx)

    # Create ROC curve:
    # we need to calculate the fpr and tpr for all thresholds of the classification
    from sklearn.metrics import roc_curve, auc
    

    # only use probs of pos. class here - 
    # since we need the prediction array to contain probability 
    # estimates of the positive class or confidence values
    # PREDICTED:
    pred_probas = model_logreg.predict_proba(test_regressors)[:,1] 
    print(pred_probas.shape) 

    # ACTUAL
    print(tdf.loc[testID,u'label'].size) 
    # print('actual test classes:',tdf.loc[testID,u'label'])
    fpr,tpr,_ = roc_curve(tdf.loc[testID,u'label'], pred_probas)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='area = %.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.title('ROC for Log. Reg. Model')
    plt.xlabel('1 - Specificity')
    plt.ylabel('Recall (i.e. Sensitivity)')

    plt.show()

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

    ## Demonstration:
    ## predict unseen and unlabelled tweets now using infer:
    ## make token lists first
    tokens = list()
    # irrelevant tests:
    tokens.append("the white house is now the insane asylum cpac".split())
    tokens.append("wow are the badgers drunj tonight".split())
    tokens.append("want to work in hampton mn view our latest opening job jobs hiring".split())
    
    # relevant tests
    tokens.append("° yesterday raining thunderstorm rn amp no school tomorrow bc of snow iowa weather wtf are you ☀️⛈❄️".split())
    tokens.append("i just love when nebraska does that really cool thing where its degrees rains and snows all in one week no big deal 🙃".split())
    tokens.append("how im waking up if theres not a snow day tomorrow".split())
    tokens.append("its coming down pretty good and its slippery out we should have a foot on the ground by morning".split())
    
    from pprint import pprint
    
    for tweet in tokens:
        
        # then make document vectors 
        # NB Won't retrieve SAME vector cf https://github.com/RaRe-Technologies/gensim/issues/374
        # "... it should wind up 'close' to the vector ..."
        dvdm = model_DM.infer_vector(tweet)     # note: may want to use many more steps than default
        dvdbow = model_DBOW.infer_vector(tweet) 

        # find the similar tweets-documents:
        sims = model_DM.docvecs.most_similar(positive=[dvdm])
        # print(sims)
        # print('debug info for inferred docvec:')
        # pprint(jlpb_classify.dump(dvdm))
        
        dv = list(dvdm) + list(dvdbow) 
        # print('dv initially')
        # pprint(jlpb_classify.dump(dv)) 
        print(dir(dv))
        print (type(dv))
        dv = np.append([1.0], dv)
        probs = model_logreg.predict_proba(dv) # to get the class probabailities
        print(probs, model_logreg.predict(dv)) # and output the class


    

    
