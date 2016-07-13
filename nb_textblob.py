# nb_textblob.py
# adapted from example tutorial:
# http://stevenloria.com/

# method A - train on the text strings we have alone (normalised)
# method B - train with unigrams
# method C - train with bigrams
# + other tries

##
# method A - train on the text strings we have alone (normalised)
##
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob

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

# get training and test data from mongodb:
if CURR_PLATFORM != 'linux':
    dbc = jlpb.get_dbc('Twitter', 'sample_a200')
    dbtest_set = jlpb.get_dbc('Twitter', 'testset_a')

else:
    dbc = jlpb.get_dbc('local', 'sample_a200')
    dbtest_set = jlpb.get_dbc('local', 'testset_a')


## TRAINING SET
train = []
# keep a list of the IDs just to double-check our test set is not identical!
train_ids = [] 

results = dbc.find()
for doc in results[:]:
    # NB or e.g.: [ ([], 'class') , ] 
    # record: nonnormalised; trigrams and bigrams joined
    train.append( (doc['txt']['normalised'], str(doc['class'])) )
    train_ids.append(doc['tweet_id'])

## TEST SET
test = []
test_ids = []

results = dbtest_set.find()
for doc in results[:]:
    test.append( (doc['txt']['normalised'], str(doc['class'])) )
    if doc['tweet_id'] in train_ids:
        print (doc['tweet_id'])
    else:
        test_ids.append(doc['tweet_id'])

#  TextBlob will treat both forms of data OK
# so, pass either string or unigrams...
cl = NaiveBayesClassifier(train)


# wraps NLTK simply: return nltk.classify.accuracy(self.classifier, test_features) 
acc = cl.accuracy(test)

print('accuracy: normalised text; train/test:', len(train), '/', len(test), '=',  acc)
cl.show_informative_features(20)
exit()
# item = item.decode('ascii', errors="replace")

## use the blob method as it is more convenient
# unicode issues?
blob = TextBlob(item)
for np in blob.noun_phrases:
    print (np)


cl.accuracy(test)

# test a new item  usage:
newitem = 'dsdsjdlaskdjkl'
cl.classify(newitem)

# top five contriobuting feats
cl.show_informative_features(5) 

# get the label probability distribution with the prob_classify(text) method.
prob_dist = cl.prob_classify(newitem)
prob_dist.max()
relevant = round(prob_dist.prob("pos"), 2)
irrelevant = round(prob_dist.prob("neg"), 2)
## 
# method B - train with unigrams
#
cl.update(new_train) # can call it like this
accuracy = cl.accuracy(test + new_test)


### can pass a custom feature-extractor function to the clasifier
## maybe try with one that removes key hashtag terms and see if it improves or not
# A feature extractor is simply a function with document (the text to extract features from)
# as the first argument.
# The function may include a second argument, train_set (the training dataset), if necessary.
#
#


# can try Noun Phrase extraction
