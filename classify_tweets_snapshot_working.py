'''
local machine test, autumn 2016. 
'''
import pandas as pd
import random
import numpy as np
import os, sys

# specify dir to save a cached Doc2Vec model later
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

# get some handy functions for plotting metrics and preprocessing text
import jlpb
import jlpb_classify


if __name__ == "__main__":
    '''
    We will use a Doc2Vec model to then train a classifier
    (Logistic Regression etc) to try to classify relevant
    tweets from irrelevant ones for topic of flooding.

    '''
    import time
    start = time.perf_counter() # keep track of processing time

    # Set up some key variables for the process of model training:
    # Use a random seed for reproducibility 
    seed = 50 

    # This can take a LOT of time if high! but should give better
    # performance for the classifier. 
    epochs = 20
    vocab_rows = 420000 # no. unlabelled tweets for building vocabulary table in Doc2Vec
    vocab_frac = 1 # when using a sample of a huge file of unlabelled tweets
    vecs = 200
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

    # tdf therefore will combine all our labelled data: i.e. 500 + 4002 labelled:
    tdf = pd.concat([zdf, xdf], axis=0)

    # make the string a list 
    # NB needed in D2V gensim: (ie split the string into words, for D2V)
    tdf.loc[:,'text'] = tdf.loc[:,'text'].map(jlpb_classify.split)
    
    print('tdf',tdf.shape)

    tdf[tdf == 'negative'] = 0
    tdf[tdf == 'positive'] = 1
    
    print('TDF tail:5', tdf.tail(5))

    # Randomise the order of the Labelled set rows 
    tdf = tdf.sample(frac=vocab_frac, random_state=np.random.RandomState(seed)).\
        reset_index(drop=True)

    print('Tail labelled set', tdf.tail(5))
    print('Dims tdf',tdf.shape)

    # Load in our unlabelled data set of tweets to build the doc2vec vocabulary table.
    udf = pd.read_csv('unlab420k.csv') #unlabelled.csv has 50k; unlab420k.csv all!
    udf = udf[[u'text']]
    
    print('randomising unlabelled tweets dataframe for vocab build ...')
    
    udf = udf.sample(frac=vocab_frac, random_state=np.random.RandomState(seed)).\
        reset_index(drop=True) 

    # uncomment to use with 23rd data only:
    # udf = udf.iloc[:vocab_rows]
    print(udf.size , ' unlabelled tweets for vocab')
    

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
    training_num = total_num - test_num
    print('Test set size', test_num, 'Training set size', training_num)
    
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

    # build fresh models if True set below!
    # (otherwise load from disk)
    most_recent = dir_model + 'Mac_d2v_win10_420kseed50_200se5_ep20_minc6'
    # save current labelled dataframe to a CSV 
    tdf.to_csv(most_recent + 'LOG.csv')

    model_DM, model_DBOW = (None, None)

    # Train the two different methods of the Doc2Vec algorithm:
    # and change below line to True to load in new models:
    if False:
        # Parameters can be adjusted to try to get better accuracy from classifier.
        model_DM = Doc2Vec(size=vecs, window=10, min_count=6, sample=1e-5,\
         negative=5, workers=cores,  dm=1, dm_concat=1 )
        model_DBOW = Doc2Vec(size=vecs, window=10, min_count=6, sample=1e-5,\
         negative=5, workers=cores, dm=0)

        # construct the vocabulary tables for our models
        model_DM.build_vocab(training_doc)
        model_DBOW.build_vocab(training_doc)
        
        # train the models themselves
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


    # viz some word vecs using tSNE
    import tsneviz

    tsneviz.display_closestwords_tsnescatterplot(model_DM, 'flooding')
    tsneviz.display_closestwords_tsnescatterplot(model_DM, 'thunderstorm')
    tsneviz.display_closestwords_tsnescatterplot(model_DM, 'delays')


    # Output some Diagnostic word vectors:
    print('similar lunch flooding', model_DM.similarity('lunch', 'flooding'))
    print('similar rain thunderstorm', model_DM.similarity('rain', 'thunderstorm'))
    print('nonmatch', model_DM.doesnt_match("delay government flooding lightning".split()))
    print('euref sim by word', model_DM.similar_by_word('euref'))
    print('umbrella emoji ☔️☔️☔️', model_DM.similar_by_word('☔️☔️☔️'))
    print('flooding ', model_DM.similar_by_word('flooding'))
    print('weather', model_DM.most_similar('weather'))
    print('rain', model_DM.most_similar('rain'))
    print('lightning', model_DM.most_similar('lightning'))
    print('thunder', model_DM.most_similar('thunder'))
    print('thunderstorm', model_DM.most_similar('thunderstorm'))
    print('ukstorm', model_DM.most_similar('ukstorm'))
    print('trains', model_DM.most_similar('trains'))
    print('delays', model_DM.most_similar('delays')) 
    print('light thunder similarity', model_DM.similarity('lightning', 'thunder'))
    
    
    '''
    Use  SciKit estimator (log. regression or others) to train classifier from Doc2Vec model
    and labelled data.

    Then output plots of confusion matrices of the accuracy of the model applied to 
    test data.

    Credit: https://www.zybuluo.com/HaomingJiang/note/462804
    NB Adapted methodology from a tutorial in Doc2Vec -- URL above -- to scaffold 
    this classification.

    '''
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    import statsmodels.api as sm
    
    random.seed(1234) 

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

    model_logreg = LogisticRegression(C=0.01, class_weight='balanced', tol=0.001, penalty='l2', max_iter=600, solver='sag', n_jobs=-1)
    print('please wait.......fitting LR model.....')
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
        # debug: print('probs', model_logreg.predict_proba([test_regressors[i]]))
        if test_predictions[i] == tdf.loc[testID[i],u'label']:
            
            # output any true positives:
            if(test_predictions[i] == 1):
                jlpb.uprint('Correct: id', str(i) + ', tdf_row:' + str(testID[i]) + ', ', \
                    str(tdf.loc[testID[i], u'label']),\
                    tdf.loc[testID[i], u'text'])
            accuracy = accuracy + 1
        else:
            jlpb.uprint('Incorrect:'+ str(i) + ', tdf_row:' + str(testID[i]) +', actual', \
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
    cast = (cast.values).astype(np.int8) # numpy.ndarray now.

    # Key result - TEST matrix:
    confusion_mtx = confusion_matrix(cast, test_predictions)
    jlpb_classify.show_confusion_matrix(confusion_mtx)


    train_predictions = model_logreg.predict(train_regressors)
    accuracy = 0
    for i in range(0,len(train_targets)):
        if train_predictions[i] == train_targets[i]:
            accuracy = accuracy + 1
    accuracies = accuracies + [1.0 * accuracy / len(train_targets)]
    confusion_mtx = confusion_matrix(train_targets, train_predictions)
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
    # Create ROC curve:
    # we need to calculate the fpr and tpr for all class probability thresholds 
    from sklearn.metrics import roc_curve, auc, precision_score, average_precision_score, \
    f1_score, precision_recall_curve
    import matplotlib.pyplot as plt

    # only use probs of pos. class here - 
    # since we need the prediction array to contain probability 
    # estimates of the positive class or confidence values
    # PREDICTED y-scores:
    pred_probas = model_logreg.predict_proba(test_regressors)[:, 1] 
    print(pred_probas.shape) 

    # ACTUAL y-test/ground truths
    print('ground truths', tdf.loc[testID,u'label'].size, ' and ', tdf.loc[testID,u'label'].dtype) 
    
    fpr, tpr, _ = roc_curve(tdf.loc[testID,u'label'], pred_probas)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr, tpr, label='Area Under ROC curve = %.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--') # indicate a randomly guessing classifier performance 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.title('ROC for Log. Reg. Model')
    plt.xlabel('1 - Specificity')
    plt.ylabel('Recall (i.e. Sensitivity)')

    plt.show()



    # average_precision = average_precision_score(tdf.loc[testID,u'label'], pred_probas)
    average_precision = average_precision_score(tdf.loc[testID,u'label'].astype(int), pred_probas)
    print('Average precision-recall score: ', average_precision)

    # print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    # other metrics
    # print("Precision: %2f" % precision_score(tdf.loc[testID,u'label'], test_predictions)
    print("F1: %2f" % f1_score(tdf.loc[testID,u'label'].astype(int), test_predictions, average="macro"))
    
    # need to just get the probs of the positive class, hence pred_probas[:,1]  
    # (NB or try instead: y_scores_lr = lr.fit(X_train, y_train).decision_function(X_test))
    precision, recall, _ = precision_recall_curve(tdf.loc[testID,u'label'], pred_probas) #[:,1]
    area = auc(recall, precision)
    print ("Area Under PR Curve(AP): %0.2f" % area)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(average_precision))
    plt.show()
    
    ## Demonstration:
    ## predict unseen and unlabelled tweets now using infer_vector():
    ## make a set of test tweets as tokenised lists first
    tokens = list()

    # irrelevant tests:
    tokens.append("what kind of sick mosquito bites the bottom of your foot".split())
    tokens.append("if there was a bucket for pins on twitter we could easily fillet".split())
    tokens.append("when people message you and go why you up late".split())
    
    # relevant tests
    tokens.append("thanks lewisham station told to take blackfriars train and use underground district and circle lines suspended tfl onlyabitofrain".split())
    tokens.append("always rely on to get into work when floods hit the underground".split())
    tokens.append("everyone is enjoying the spa garden sunshine including the fish the moorhens ducks the turtles or is that tortoises".split())
    tokens.append("massive amount of rain in a short time and small old drains in the road couldnt handle it".split())
    
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
        #print(dir(dv))
        #print (type(dv))
        dv = np.append([1.0], dv)
        probs = model_logreg.predict_proba([dv]) # to get the class probabailities
        print(probs, model_logreg.predict([dv])) # and output the class

   


    

    
