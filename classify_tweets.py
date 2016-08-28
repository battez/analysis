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


import time
start = time.perf_counter()


def split(text):
    '''
    text string; a tweet (already tokenised)
    returns list; words
    '''
    words = text.split(' ')
    return words


def show_confusion_matrix(C,class_labs=['0','1']):
    '''
    credit: http://notmatthancock.github.io/2015/10/28/confusion-matrix.html
    Pretty-plot scikit learn confusion matrices nicely:
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labs: list of strings, default simply labels 0 and 1.

    Draws confusion matrix with associated metrics.
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    
    assert C.shape == (2,2), "Confusion matrix valid from binary classification only."
    
    # true negative, false positive, etc...
    tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];

    NP = fn+tp # Num positive examples
    NN = tn+fp # Num negative examples
    N  = NP+NN

    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(2.5,-0.5)
    ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)
    ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)
    ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)
    ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(class_labs + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34,1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labs + [''],rotation=90)
    ax.set_yticks([0,1,2])
    ax.yaxis.set_label_coords(-0.09,0.65)


    # Fill in initial metrics: tp, tn, etc...
    ax.text(0,0,
            'True Neg: %d\n(Num Neg: %d)'%(tn,NN),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,1,
            'False Neg: %d'%fn,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,0,
            'False Pos: %d'%fp,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    ax.text(1,1,
            'True Pos: %d\n(Num Pos: %d)'%(tp,NP),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2,0,
            'False Pos Rate: %.2f'%(fp / (fp+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,1,
            'True Pos Rate: %.2f'%(tp / (tp+fn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,2,
            'Accuracy: %.2f'%((tp+tn+0.)/N),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,2,
            'Neg Pre Val: %.2f'%(1-fn/(fn+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,2,
            'Pos Pred Val: %.2f'%(tp/(tp+fp+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    plt.tight_layout()
    plt.show()


'''
We will use a Doc2Vec model to then train a classifier
(Logistic Regression currently) to try to classify relevant
tweets from irrelevant ones for topic of flooding.

'''
# Set up some key variables for the process of model training:
# Use a random seed for reproducibility 
seed = 0
seed = 99 
seed = 20
seed = 40 
# This can take a LOT of time if high! but should give better
# performance for the classifier. 
epochs = 16 


##
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
tdf.loc[:,'text'] = tdf.loc[:,'text'].map(split)

print('head:20', tdf.head(20))
print('tail:20', tdf.tail(20))
print('tdf',tdf.shape)

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
udf = udf.iloc[:50000]
udf.loc[:,'text'] = udf.loc[:,'text'].map(split)
#print('Tail Unlabelled set',  udf.tail())

# Reference: https://radimrehurek.com/gensim/models/doc2vec.html 
# Gensim Doc2Vec for high-dim vectors in model(s) for each tweet:
from gensim.models.doc2vec import TaggedDocument, Doc2Vec


total_num = int(tdf.size/2)
print(total_num)
print(tdf.size)
total_num_unlabelled = udf.size

# split for the needed test and training data 
# maintain approx. a 9:1 ratio of training:test,
# as we have relatively little labelled data.
test_num = 450
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

print('tail 20 classLabels', class_labels.tail(20))
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
    model_DM = Doc2Vec(size=152, window=5, min_count=1, sample=1e-4,\
     negative=5, workers=cores,  dm=1, dm_concat=1 )
    model_DBOW = Doc2Vec(size=152, window=5, min_count=1, sample=1e-4,\
     negative=5, workers=cores, dm=0)

    # construct the vocabs for our models
    model_DM.build_vocab(training_doc)
    model_DBOW.build_vocab(training_doc)

    fout = 'c1200_50kseed40_16ep152se4DM.d2v'
    model_DM.save(most_recent + fout)

    fout = 'c1200_50kseed40_16ep152se4DBOW.d2v'
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
new_index = random.sample(range(0,total_num),total_num)

# set the IDs for the test set:
testID = new_index[-test_num:]

# set the IDs for the training set
trainID = new_index[:-test_num]

train_targets, train_regressors = zip(*[(class_labels[id], \
    list(model_DM.docvecs[id]) + list(model_DBOW.docvecs[id])) for id in trainID])
train_regressors = sm.add_constant(train_regressors)

# Uncomment to use this for multiclass log. regression:
# Todo: add class in dataset for brexit topic
# model_logreg = LogisticRegression(multi_class='multinomial',solver='lbfgs')

# Train a two-class log. regression classifier, and use
# scikit parameter to adjust foru our imbalanced dataset:
# class_weight="balanced", 
# use default L2 penalty, tolerance speeds training time, 
model_logreg = LogisticRegression(C=1200, penalty='l2', tol=0.0001, n_jobs=-1)
model_logreg.fit(train_regressors, train_targets) # first is x-axis, targets the y

## When ready: use to save a nice model:
from sklearn.externals import joblib
dir_model = 'C:/Users/johnbarker/Downloads/'
filename_model = 'logreg_model.pkl'
joblib.dump(model_logreg, dir_model + filename_model)  # try compress

# When needed, test reloading with
#  = joblib.load(dir_model + filename_model) 


accuracies = []
test_regressors = [list(model_DM.docvecs[id])+list(model_DBOW.docvecs[id]) for id in testID]
test_regressors = sm.add_constant(test_regressors)
test_predictions = model_logreg.predict(test_regressors)
accuracy = 0

# print('unseen predictions:', unseen_predictions)
for i in range(0, test_num):
    if test_predictions[i] == tdf.loc[testID[i],u'label']:
        
        if(test_predictions[i] == 'positive'):
            jlpb.uprint('Correct: id', str(i) + '/' + str(testID[i]) + ', ', \
                tdf.loc[testID[i], u'label'],\
                tdf.loc[testID[i], u'text'])
        accuracy = accuracy + 1
    else:
        jlpb.uprint('WRONG:'+ str(i) + '/' + str(testID[i]) +', should be', \
            tdf.loc[testID[i], u'label'], tdf.loc[testID[i], u'text'])

# calculate the final accuracy:        
accuracies = accuracies + [1.0 * accuracy / test_num]

## Show user time needed for this classifier:
total = int((time.perf_counter() - start) / 60)
print("Process took %s minutes" % total)

# OUTPUT Accuracy rates and so on:
#
# Produce some confusion matrices and plot them:
labels = ['negative', 'positive']

confusion_mtx = confusion_matrix(test_predictions,(tdf.loc[testID,u'label']))
print('test conf matrix: ', confusion_mtx)
show_confusion_matrix(confusion_mtx)


train_predictions = model_logreg.predict(train_regressors)
accuracy = 0
for i in range(0,len(train_targets)):
    if train_predictions[i] == train_targets[i]:
        accuracy = accuracy + 1
accuracies = accuracies + [1.0 * accuracy / len(train_targets)]
confusion_mtx = confusion_matrix(train_predictions,train_targets)
print('training conf matrix: ', confusion_mtx)

show_confusion_matrix(confusion_mtx)


## Todo:
## predict unseen and unlabelled tweets
##
## make token lists first
# then make document vectors 
# then model_logreg.predict() / 
# model_logreg.predict_proba() to get the class probabailities
