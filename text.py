
'''
baseline a Random Forest Classifier on tweets:
'''
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

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

# training / validation cutoff split:
cutoff = 720

# get training and test data from mongodb:
if CURR_PLATFORM != 'linux':
    dbc = jlpb.get_dbc('Twitter', 'stream2flood_all')
    dbtest_set = jlpb.get_dbc('Twitter', 'testset_a')

else:
    dbc = jlpb.get_dbc('local', 'sample_a200')
    dbtest_set = jlpb.get_dbc('local', 'testset_a')
# should have already preprocessed text and have unigrams
vec = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 10000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
results = dbc.find({}, \
    { 'txt.parsed': 1, 't_class': 1, 'txt.normalised':1, '_id':0 })\
    .sort([("timestamp_ms",pymongo.DESCENDING)])
tweets, txt, labels = ([],[],[])

def removeTerm(terms, clean_me):
    for idx, word in enumerate(clean_me):
        if word in terms:
            del clean_me[idx]
    return clean_me 

terms = ['flood','floods','flooded','flooding','flooder']

# load in the collection which will be split at cutoff into
# training and test sets:
for tweet in results:
    txt.append(tweet['txt']['normalised'])
    labels.append(tweet['t_class'])
    parsed = removeTerm(terms, tweet['txt']['parsed'])
    tweets.append(' '.join(parsed))

train_data_features = vec.fit_transform(tweets[:cutoff])

# numpy style array
train_data_features = train_data_features.toarray()

# dims
print ('train dims:', train_data_features.shape)

vocabulary = vec.get_feature_names()
# print (vocabulary)

# freq. counts of each word
dist = np.sum(train_data_features, axis=0)

# print word and freq in our training set
for tag, count in zip(vocabulary, dist):
    # print (count, tag)
    continue

print("random forest ########################################################")
from sklearn.ensemble import RandomForestClassifier

# Random Forest classifier; n - trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit RF to train set, feats = BOW 
# ...t_class is our relevance label 0 or 1
forest = forest.fit(train_data_features, labels[:cutoff])

print('RF trained ###########################################################')

# Get BOW for test set; convert to numpy array
test_data_features = vec.transform(tweets[cutoff:])
test_data_features = test_data_features.toarray()

print ('test dims:', test_data_features.shape)

res = forest.predict(test_data_features)
output = pd.DataFrame(data={"text": txt[cutoff:], "class_label": res})
jlpb.uprint(output)
output.to_csv( "RF_Bag_Words_model" + str(cutoff) + ".csv", index=False, quoting=3 )
