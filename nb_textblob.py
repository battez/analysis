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


def resolve_redirect(url):
    '''
    follow redirects until there isn't any more redirects,
    works better than below fn!
    '''
    import requests
    # verify can be set to False for more laxity but only if needed. And Insecure!
    try:
        req = requests.head(url, allow_redirects=True, verify=True) 

    except Exception:
        return False
    
    return req.url


def resolve_http_redirect(url, depth=0):
    '''
    -- Not using at present as did not work as well as above fn --
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


def summarise_links(dbc):
    '''
    Populate a mongo collection with article summaries for 
    its tweets' URL entities.

    Fixme: at the moment only uses the first URL for that tweet,
    as any second URLs often just for a 3rd party app. But could 
    be extended.
    '''
    import summarise
    
    # get all tweets with url entities
    query = {'entities.urls.0.expanded_url':{"$exists":True}, 'url':{'$exists':False}}
    results = dbc.find(query)
   
    for doc in results[:]:
        
        # returns: title, keywords, summary, top_img_src
        resolved_url = resolve_redirect(doc['entities']['urls'][0]['expanded_url'])
        if resolved_url == False:
            print('redirection problem: ', doc['entities']['urls'][0]['expanded_url'])
            continue
        try:
            summary = summarise.summarise_one(resolved_url, \
            True, True, True, False)

        except Exception as e:
            print('link summarise had error: ', e)
            continue
    
        # save the results back to that tweet
        if type(summary[1]) is bool or not summary[1]:
            continue
        else:
            dbc.update({'_id':doc['_id']}, {\
            '$push':{'url.keywords': {'$each':summary[1]}},\
            '$set':{'url.title':summary[0],'url.summary':summary[2]} })

def summarise_instagram(dbc):
    '''
    Populate a mongo collection with image summaries for 
    its tweets' Instagram re-posts .
    these can be found with a query in mongoDB:
    {'entities.urls':{$ne:[]}, 'entities.urls.0.display_url':/^instagram.com/}

    We first get the URLs, then get the top image from that URL using newspaper.
    then we use this to get the image. 
    '''
    import summarise
    import re

    # get all tweets with instagram url 
    regex = re.compile('^instagram.com')
    query = {'img':{'$exists':False}, 'entities.urls.0.display_url':{'$regex':regex}}
    results = dbc.find(query)
    
    for doc in results:
        
        # returns: title, keywords, summary, top_img_src
        resolved_url = doc['entities']['urls'][0]['expanded_url']
        
        try:
            summary = summarise.get_top_img(resolved_url)
        except Exception as e:
            print('link summarise had error: ', e)
            continue

        print(summary)
        
        # save the results back to that tweet
        if not summary:
            continue
            
        else:
            # have the URL of img, so now classify it:

            pass


def summarise_images(dbc, options={'threshold':0.15}, watson=False ):
    '''
    Populate a mongo collection with image summaries for 
    its tweets' media_url image entities.
    '''
    import summarise
    count = 0 # keep track of how many records we update

    img_key = 'img'
    if watson:
        img_key = 'img_watson'

    # get all tweets with url entities, but not had this img classing done before:
    query = {'entities.media.0.media_url':{'$exists':True}, img_key:{'$exists':False}}
    results = dbc.find(query, no_cursor_timeout=True)
    
    for doc in results:
        
        # returns: title, keywords, summary, top_img_src
        resolved_url = resolve_redirect(doc['entities']['media'][0]['media_url'])
        print('trying url...', doc['entities']['media'][0]['media_url'])

        if resolved_url == False:
            print('redirection problem: ', doc['entities']['media'][0]['media_url'])
            continue

        data = {}

        # which method to use:
        if not watson:
            try:
                output = summarise.summarise_img(resolved_url, options)
                summary = output.result

            except Exception as e:
                print('summary failed:', e)
                continue
        
            # save the results back to that tweet
            if not summary or ('general' not in summary):
                print('no web service classified image data in response')
                continue
            else:
                data = {img_key + '.keywords': summary['general']}
               
        else:
            try:
                output = summarise.summarise_watson_img(resolved_url)
                summary = output['images'][0]


            except Exception as e:
                print('summary failed:', e)
                continue
        
            # save the results back to that tweet
            print('summary', summary)
            if 'error' in summary:
                print('no web service classified image data in response')
                continue
            else:
                data = {img_key: summary['classifiers'][0]['classes']}
        
        # if we made it this far, update the db record.
        dbc.update({'_id':doc['_id']}, {'$set':data })
        count += 1

    del results
    return count


if __name__ == '__main__':   
    '''
    Run various summaries and then do a classification on twitter data.
    Uncomment below for URL extraction of summary on MongodDB
    collection
    '''
    # summarise_links(jlpb.get_dbc('Twitter', 'testset_a'))

    # image summaries - using a web service
    # options -- set threshold at 0.35
    # we only want the 'general' to be stored 
    # store just the keywords 
    coll = 'stream2_storm_all'
    summarise_instagram(jlpb.get_dbc('Twitter', coll))
    exit()
    print('running Algo summarising')
    count = summarise_images(jlpb.get_dbc('Twitter', coll), watson=False)
    print(count, 'updates')
    print('running Watson summarising')
    count = summarise_images(jlpb.get_dbc('Twitter', coll), watson=True)
    print(count, 'updates')

    from time import sleep
    sleep(1)
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
