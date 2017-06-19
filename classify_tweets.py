'''
switching in our own tweet data to Doc2Vec example as per here
https://www.zybuluo.com/HaomingJiang/note/462804
'''
import pandas as pd
import random
import numpy as np
import sys

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
    seed = 0
    seed = 99 
    seed = 20
    seed = 40 
    # This can take a LOT of time if high! but should give better
    # performance for the classifier. 
    epochs = 40
    vocab_rows = 40000 # how many tweets to use for building vocab in D2Vec
    vecs = 160
    test_num = 450

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
    print('head:20', tdf.head(20))
    print('tail:20', tdf.tail(20))
    # Randomise the order of the Labelled set rows 
    # (using a seed for reproducibility)
    tdf = tdf.sample(frac=1, random_state=np.random.RandomState(seed)).\
        reset_index(drop=True)

    print('Head labelled set', tdf.head(30))
    print('Tail labelled set', tdf.tail(30))
    print('Dims tdf',tdf.shape)

    # Load in our unlabelled data set of tweets to build the d2v vocabulary.
    udf = pd.read_csv('unlabelled.csv')
    udf = udf[[u'text']]
    udf = udf.iloc[:vocab_rows]
    udf.loc[:,'text'] = udf.loc[:,'text'].map(jlpb_classify.split)
    #print('Tail Unlabelled set',  udf.tail())

    # Reference: https://radimrehurek.com/gensim/models/doc2vec.html 
    # Gensim Doc2Vec for high-dim vectors in model(s) for each tweet:
    from gensim.models.doc2vec import TaggedDocument, Doc2Vec

    total_num = int(tdf.size/2)
    print('tweets data dims: ', total_num)
    print(tdf.size)
    total_num_unlabelled = udf.size

    # split for the needed test and training data 
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

    # find out max system workers from the cores
    # so we can make use of the max CPU:
    import multiprocessing
    cores = multiprocessing.cpu_count()

    # build fresh doc2vec models if True set below!
    # (otherwise load from disk)
    most_recent = 'd2v_mod_win5_'
    model_DM, model_DBOW = (None, None)

    # change below line to True to load in new models:
    if True:
        # Parameters can be adjusted to try to get better accuracy from classifier.
        model_DM = Doc2Vec(size=vecs, window=5, min_count=1, sample=1e-4,\
         negative=5, workers=cores,  dm=1, dm_concat=1 )
        model_DBOW = Doc2Vec(size=vecs, window=5, min_count=1, sample=1e-4,\
         negative=5, workers=cores, dm=0)

        # construct the vocabs for our models
        model_DM.build_vocab(training_doc)
        model_DBOW.build_vocab(training_doc)

        fout = 'c1200_50kseed40_40ep160se4DM.d2v'
        model_DM.save(most_recent + fout)

        fout = 'c1200_50kseed40_40ep160se4DBOW.d2v'
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

    for it in range(0,epochs):
        # progress as this takes a long time:
        if (it % 10) == 0:
            print('epoch ' + str(it) + ' of ' + str(epochs))

        random.shuffle(doc2vec_train_id)
        training_doc = [documents_all[id] for id in doc2vec_train_id]
        model_DM.train(training_doc)
        model_DBOW.train(training_doc)

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


    random.seed(1212)
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

    # Uncomment to use this for multiclass log. regression:
    # Todo: add third class in dataset for brexit topic
    # model_logreg = LogisticRegression(multi_class='multinomial',solver='lbfgs')

    # Train a two-class log. regression classifier, and use
    # scikit parameter to adjust for our imbalanced dataset:
    # class_weight="balanced", 
    # use default L2 penalty, tolerance speeds training time, 
    # model_logreg = LogisticRegression(C=1200, penalty='l2', tol=0.0001, n_jobs=-1)
    model_logreg = LogisticRegression(C=1200, penalty='l2', tol=0.0001, n_jobs=-1)

    model_logreg.fit(train_regressors, train_targets) # first is x-axis, targets the y

    ## SAVE MODEL TO PICKLE etc ======================================================
    ## When ready: use to save a nice model:
    from sklearn.externals import joblib
    dir_model = 'C:/Users/johnbarker/Downloads/'
    filename_model = 'logreg_model.pkl'
    joblib.dump(model_logreg, dir_model + filename_model)  # try compress

    # When needed, test reloading with
    #  = joblib.load(dir_model + filename_model) 

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
                jlpb.uprint('Correct: id', str(i) + '/' + str(testID[i]) + ', ', \
                    str(tdf.loc[testID[i], u'label']),\
                    tdf.loc[testID[i], u'text'])
            accuracy = accuracy + 1
        else:
            jlpb.uprint('WRONG:'+ str(i) + '/' + str(testID[i]) +', should be', \
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
    ## make token lists first
    # then make document vectors 
    # then model_logreg.predict() / 
    # model_logreg.predict_proba() to get the class probabailities


    
    # CREDIT: Use library from (something amiss with my parameters! Needs work..)
    # https://github.com/tmadl/highdimensional-decision-boundary-plot
    # prototype code for visualising high-dim data from LogReg Model.
    from decisionboundaryplot import DBPlot
    import matplotlib.pyplot as plt
    from sklearn.learning_curve import learning_curve
    from numpy.random.mtrand import permutation

    # plot high-dimensional decision boundary
    db = DBPlot(model_logreg)
    X = test_regressors
    print('X: ', X)
    y = cast

    db.fit(X, y, training_indices=0.5)
    db.plot(plt, generate_testpoints=True)  # set generate_testpoints=False to speed up plotting
    plt.show()

    # plot learning curves for comparison
    N = 10
    train_sizes, train_scores, test_scores = learning_curve(
        model_logreg, X, y, cv=5, train_sizes=np.linspace(.2, 1.0, N))

    plt.errorbar(train_sizes, np.mean(train_scores, axis=1),
                 np.std(train_scores, axis=1) / np.sqrt(N))
    plt.errorbar(train_sizes, np.mean(test_scores, axis=1),
                 np.std(test_scores, axis=1) / np.sqrt(N), c='r')

    plt.legend(["Accuracies on training set", "Accuracies on test set"])
    plt.xlabel("Number of data points")
    plt.title(str(model_logreg))
    plt.show()
    
