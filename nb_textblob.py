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
from textblob.classifiers import NaiveBayesClassifier, DecisionTreeClassifier
from textblob import TextBlob
from nltk import tokenize 

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


def join_ngrams(feats):
    return [' '.join(item) for item in feats]






def resolve_http_redirect(url, depth=0):
    '''
    Recursively follow redirects until there isn't a location header
    Credit: http://www.zacwitte.com/resolving-http-redirects-in-python
    Jlpb updates for py3k
    '''
    from urllib.parse import urlparse
    import http.client

    if depth > 10:
        raise Exception("Redirected "+depth+" times, giving up.")
    o = urlparse(url,allow_fragments=True)
    conn = http.client.HTTPConnection(o.netloc)
    path = o.path
    if o.query:
        path +='?'+o.query
    conn.request("HEAD", path)
    res = conn.getresponse()
    headers = dict(res.getheaders())
    if 'location' in headers and headers['location'] != url:
        return resolve_http_redirect(headers['location'], depth+1)
    else:
        return url


# This is for Py2k.  For Py3k, use http.client and urllib.parse instead, and
# use // instead of / for the division




def summarise_links(dbc):
    '''
    Populate a mongo collection with article summaries for 
    its tweets' URL entities.

    Fixme: at the moment only uses the first URL for that tweet,
    as any second URLs often just for a thrid party app. But could 
    be improved.
    '''
    import summarise
    
    # get all tweets with url entities
    
    query = {'entities.urls.0.expanded_url':{"$exists":True}}
    results = dbc.find(query)
   
    
    
    for doc in results:
        
        # returns: title, keywords, summary, top_img_src
        resolved_url = resolve_http_redirect(doc['entities']['urls'][0]['expanded_url'], 0)
        print(resolved_url)
        summary = summarise.summarise_one(resolved_url, \
         True, True, True, False)
        print(summary)

        print('done')
        exit('first')

        # save the results back to that tweet
        dbc.update({'_id':doc['_id']}, {\
        '$push':{'url.keywords': {'$each':summary[1]}},\
        '$set':{'url.title':summary[0],'url.summary':summary[2]} })

summarise_links(jlpb.get_dbc('Twitter', 'stream2flood_all'))

exit('completed summaries')
# what is the feature ? record: nonnormalised; trigrams and bigrams joined
params = [('original','text'), ('txt','normalised'), ('txt','parsed')]
# params = [('txt','bigrams')]
for param in params:

    ## TRAINING SET -------------
    train = []
    results = dbc.find()
    for doc in results[:]:
        # NB or e.g.: [ ([], 'class') , ]
        if(param[1] not in ['bigrams','trigrams']):
            train.append( (doc[param[0]][param[1]], str(doc['class'])) )
        else: 
            # join the ngrams together so we can use them
            ngrams = join_ngrams(doc[param[0]][param[1]])
            train.append( (ngrams, str(doc['class'])) )


        
    ## TEST SET -----------------
    test = []
    results = dbtest_set.find({'class':{'$ne':1}})  # {'class':{'$eq':1}}
    for doc in results:
        if(param[1] not in ['bigrams','trigrams']):
            test.append( (doc[param[0]][param[1]], str(doc['class'])) )
        else:
            # join the ngrams together so we can use them
            ngrams = join_ngrams(doc[param[0]][param[1]])
            test.append( (ngrams, str(doc['class'])) )


    cl = DecisionTreeClassifier(train)
    type = 'DecisionTree'
    # cl = NaiveBayesClassifier(train)
    # type = 'NaiveBayes'

    # wraps NLTK simply: return nltk.classify.accuracy(self.classifier, test_features) 
    acc = cl.accuracy(test) * 100
    print('Classifier Type      | ', type, ' with ', '.'.join(param))
    print('Accuracy, train/test | ', '=',  str(acc), '% ,', len(train), '/', len(test))
    #cl.show_informative_features(30)
    print ('\n')
    print ('\n')


# item = item.decode('ascii', errors="replace")
exit('')
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
