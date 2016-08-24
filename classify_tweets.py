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

seed = 0
seed = 99 # for reproducibility 
seed = 20

def split(text):
    '''
    :text : string; a tweet (already tokenised)
    :return : list; words
    '''
    words = text.split(' ')

    return words


def show_confusion_matrix(C,class_labels=['0','1']):
    '''
    credit: http://notmatthancock.github.io/2015/10/28/confusion-matrix.html
    Pretty-plot scikit learn confusion matrices nicely:
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.

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
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34,1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''],rotation=90)
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



df = pd.read_csv('pos_frm500.csv')
df = df[[u'label',u'text']]
print(df.tail())

ndf = pd.read_csv('neg_frm500.csv')
ndf = ndf[[u'label',u'text']]
print(ndf.tail())

## join these two: 
zdf = pd.concat([df, ndf], axis=0)


# trying this 4002 row sized labelled set instead:
xdf = pd.read_csv('tokens_buffer10k4002set.csv')
xdf = xdf[[u'label',u'text']]

# mix 500 + 4002 labelled:
tdf = pd.concat([zdf, xdf], axis=0)
# make the string a list (ie split the string into words, for D2V)
tdf.loc[:,'text'] = tdf.loc[:,'text'].map(split)

print(tdf.tail())
print('tdf',tdf.shape)

# Randomise the order of the Labelled set rows (using a seed for reproudicibility)
tdf = tdf.sample(frac=1, random_state=np.random.RandomState(seed)).\
    reset_index(drop=True)

print('Tail labelled set', tdf.tail())
print('Dims tdf',tdf.shape)

udf = pd.read_csv('unlabelled.csv')
udf = udf[[u'text']]
udf = udf.iloc[:30000]
udf.loc[:,'text'] = udf.loc[:,'text'].map(split)
print('Tail Unlabelled set',  udf.tail())

# Doc2Vec for high-dim vectors in model(s) for each tweet:
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

total_num = int(tdf.size/2)
print(total_num)
print(tdf.size)
total_num_unlabelled = udf.size

# split for the needed test and training data 
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

# training documents 
training_doc = [documents_all[id] for id in doc2vec_train_id]

# class labels 
class_labels = tdf.loc[:,'label']
print('All class labels:', set(class_labels))

# find out max system workers from the cores:
import multiprocessing
cores = multiprocessing.cpu_count()

# build doc2vec models if True!
most_recent = 'd2v_mod_win10_'
model_DM, model_DBOW = (None, None)

# change below line to True to load in new models:
if True:
    model_DM = Doc2Vec(size=300, window=10, min_count=1, sample=1e-4,\
     negative=5, workers=cores,  dm=1, dm_concat=1 )
    model_DBOW = Doc2Vec(size=300, window=10, min_count=1, sample=1e-4,\
     negative=5, workers=cores, dm=0)

    # construct the vocabs for our models
    model_DM.build_vocab(training_doc)
    model_DBOW.build_vocab(training_doc)

    fout = '300w8se4DM.d2v'
    model_DM.save(most_recent + fout)

    fout = '300w8se4DBOW.d2v'
    model_DBOW.save(most_recent + fout)

else:
    fout = 'DM.d2v'
    model_DM = Doc2Vec.load(most_recent + fout)

    fout = 'DBOW.d2v'
    model_DBOW = Doc2Vec.load(most_recent + fout)


# train the two different methods of the Doc2Vec algorithm:
# NB DBOW is more similar to the recommended skip-gram of 
# Word2Vec by the original paper's authors.  
for it in range(0,20):
    random.shuffle(doc2vec_train_id)
    training_doc = [documents_all[id] for id in doc2vec_train_id]
    model_DM.train(training_doc)
    model_DBOW.train(training_doc)

'''
Use Logistic Regression and train a classifier from Doc2Vec model and labelled data.

Then output plots of confusion matrices of the accuracy of the model applied to 
test data.
'''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
random.seed(1212)
new_index = random.sample(range(0,total_num),total_num)
testID = new_index[-test_num:]
trainID = new_index[:-test_num]
train_targets, train_regressors = zip(*[(class_labels[id], \
    list(model_DM.docvecs[id]) + list(model_DBOW.docvecs[id])) for id in trainID])
train_regressors = sm.add_constant(train_regressors)

# Uncomment to use this for multiclass log. regression:
# Todo: add class in dataset for brexit topic
# model_logreg = LogisticRegression(multi_class='multinomial',solver='lbfgs')

# Train a two-class log. regression classifier, and use
# scikit parameter to adjust foru our imbalanced dataset:
model_logreg = LogisticRegression(class_weight="balanced", n_jobs=-1)
model_logreg.fit(train_regressors, train_targets) # first is x-axis, targets the y

## When ready: use to save a nice model:
from sklearn.externals import joblib
dir_model = 'C:/Users/johnbarker/Downloads/'
filename_model = 'logreg_model.pkl'
joblib.dump(model_logreg, dir_model + filename_model, compress=9)  # try compress

# When needed, test reloading with
#  = joblib.load(dir_model + filename_model) 


accuracies = []
test_regressors = [list(model_DM.docvecs[id])+list(model_DBOW.docvecs[id]) for id in testID]
test_regressors = sm.add_constant(test_regressors)
test_predictions = model_logreg.predict(test_regressors)
accuracy = 0

for i in range(0, test_num):
    if test_predictions[i] == tdf.loc[testID[i],u'label']:
        if(test_predictions[i] == 'positive'):
            jlpb.uprint('Correct: ', tdf.loc[testID[i], u'label'],\
             tdf.loc[testID[i], u'text'])
        accuracy = accuracy + 1
    else:
        jlpb.uprint('WRONG: should be', tdf.loc[testID[i], u'label'], tdf.loc[testID[i], u'text'])
accuracies = accuracies + [1.0 * accuracy / test_num]

# Produce some confusion matrices and plot them:
labels = ['negative', 'positive']
confusion_mtx = confusion_matrix(test_predictions,(tdf.loc[testID,u'label']), labels)
show_confusion_matrix(confusion_mtx, labels)


train_predictions = model_logreg.predict(train_regressors)
accuracy = 0
for i in range(0,len(train_targets)):
    if train_predictions[i] == train_targets[i]:
        accuracy = accuracy + 1
accuracies = accuracies + [1.0 * accuracy / len(train_targets)]
confusion_mtx = confusion_matrix(train_predictions,train_targets, labels)
show_confusion_matrix(confusion_mtx, labels)
