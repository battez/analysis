'''
switching in our own tweet data to Doc2Vec example as per here
https://www.zybuluo.com/HaomingJiang/note/462804
'''
import pandas as pd
import random
import numpy as np

seed = 0 # for reproducibility 

def split(text):
    '''
    :text : string; a tweet (already tokenised)
    :return : list; words
    '''
    words = text.split(' ')

    return words


df = pd.read_csv('train_pos_frm500.csv')
df = df[[u'label',u'text']]
print(df.tail())

ndf = pd.read_csv('train_neg_frm500.csv')
ndf = ndf[[u'label',u'text']]
print(ndf.tail())

if isinstance(df, pd.DataFrame):
    tdf = pd.concat([df, ndf], axis=0)

tdf.loc[:,'text'] = tdf.loc[:,'text'].map(split)

print(tdf.tail())
print('tdf',tdf.shape)
tdf = tdf.sample(frac=1, random_state=np.random.RandomState(seed)).reset_index(drop=True)

print(tdf.tail())
print('tdf',tdf.shape)

udf = pd.read_csv('unlabelled10k.csv')
udf = udf[[u'text']]
udf.loc[:,'text'] = udf.loc[:,'text'].map(split)
print(udf.tail())


from gensim.models.doc2vec import TaggedDocument, Doc2Vec

TotalNum = int(tdf.size/2)
print(TotalNum)
print(tdf.size)
TotalNum_Unlabed = udf.size

# split for the needed test and training data 
TestNum =100
print('test set size', TestNum)
TrainNum=TotalNum-TestNum
print('traini set size', TrainNum)


documents = [TaggedDocument(list(tdf.loc[i,'text']),[i]) for i in range(0,TotalNum)]
documents_unlabeled = [TaggedDocument(list(udf.loc[i,'text']),[i+TotalNum]) for i in range(0,TotalNum_Unlabed)]
documents_all = documents+documents_unlabeled
Doc2VecTrainID = list(range(0,TotalNum+TotalNum_Unlabed))
random.shuffle(Doc2VecTrainID)

# training documents 
trainDoc = [documents_all[id] for id in Doc2VecTrainID]

# class labels 
Labels = tdf.loc[:,'label']

# find out max ssystem workers from the cores:
import multiprocessing
cores = multiprocessing.cpu_count()

# build doc2vec models if True!
most_recent = 'd2v_model_'
model_DM, model_DBOW = (None, None)
if False:
    model_DM = Doc2Vec(size=400, window=8, min_count=1, sample=1e-4, negative=5, workers=cores,  dm=1, dm_concat=1 )
    model_DBOW = Doc2Vec(size=400, window=8, min_count=1, sample=1e-4, negative=5, workers=cores, dm=0)

    # construct the vocabs for our models
    model_DM.build_vocab(trainDoc)
    model_DBOW.build_vocab(trainDoc)

    
    fout = 'DM.d2v'
    model_DM.save(most_recent + fout)

    fout = 'DBOW.d2v'
    model_DBOW.save(most_recent + fout)

else:
    fout = 'DM.d2v'
    model_DM = Doc2Vec.load(most_recent + fout)
    fout = 'DBOW.d2v'
    model_DBOW = Doc2Vec.load(most_recent + fout)


for it in range(0,10):
    random.shuffle(Doc2VecTrainID)
    trainDoc = [documents_all[id] for id in Doc2VecTrainID]
    model_DM.train(trainDoc)
    model_DBOW.train(trainDoc)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
random.seed(1212)
newindex = random.sample(range(0,TotalNum),TotalNum)
testID = newindex[-TestNum:]
trainID = newindex[:-TestNum]
train_targets, train_regressors = zip(*[(Labels[id], list(model_DM.docvecs[id])+list(model_DBOW.docvecs[id])) for id in trainID])
train_regressors = sm.add_constant(train_regressors)
predictor = LogisticRegression(multi_class='multinomial',solver='lbfgs')
predictor.fit(train_regressors,train_targets)



accus=[]
Gmeans=[]
test_regressors = [list(model_DM.docvecs[id])+list(model_DBOW.docvecs[id]) for id in testID]
test_regressors = sm.add_constant(test_regressors)
test_predictions = predictor.predict(test_regressors)
accu=0
for i in range(0,TestNum):
    if test_predictions[i]==tdf.loc[testID[i],u'text']:
        accu=accu+1
accus=accus+[1.0*accu/TestNum]
confusionM = confusion_matrix(test_predictions,(tdf.loc[testID,u'text']))
Gmeans=Gmeans+[pow(((1.0*confusionM[0,0]/(confusionM[1,0]+confusionM[2,0]+confusionM[0,0]))*(1.0*confusionM[1,1]/(confusionM[1,1]+confusionM[2,1]+confusionM[0,1]))*(1.0*confusionM[2,2]/(confusionM[1,2]+confusionM[2,2]+confusionM[0,2]))), 1.0/3)]
train_predictions = predictor.predict(train_regressors)
accu=0
for i in range(0,len(train_targets)):
    if train_predictions[i]==train_targets[i]:
        accu=accu+1
accus=accus+[1.0*accu/len(train_targets)]
confusionM = confusion_matrix(train_predictions,train_targets)
Gmeans=Gmeans+[pow(((1.0*confusionM[0,0]/(confusionM[1,0]+confusionM[2,0]+confusionM[0,0]))*(1.0*confusionM[1,1]/(confusionM[1,1]+confusionM[2,1]+confusionM[0,1]))*(1.0*confusionM[2,2]/(confusionM[1,2]+confusionM[2,2]+confusionM[0,2]))), 1.0/3)]
