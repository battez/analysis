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
import summarise

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
    # verify can set to False for laxity but only if needed. And Insecure!
    try:
        req = requests.head(url, allow_redirects=True, verify=True) 

    except Exception:
        return False
    
    return req.url


def summarise_links(dbc):
    '''
    Populate a mongo collection with article summaries for 
    its tweets' URL entities.

    Fixme: at the moment only uses the first URL for that tweet,
    as any second URLs often just for a 3rd party app. But could 
    be extended.
    '''    
    # get all tweets with url entities
    query = {'original.entities.urls.0.expanded_url': \
    {"$exists":True}, 'url':{'$exists':False}}
    results = dbc.find(query, no_cursor_timeout=True ).\
    sort([("_id", pymongo.DESCENDING)])
   
    for doc in results:
        
        # returns: title, keywords, summary, top_img_src
        resolved_url = resolve_redirect(\
            doc['original']['entities']['urls'][0]['expanded_url'])
        if resolved_url == False:
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

    print('links summarised - completed.')


def summarise_instagram(dbc):
    '''
    Populate a mongo collection with image summaries for 
    its tweets' Instagram re-posts .
    these can be found with a query in mongoDB:
    {'entities.urls':{$ne:[]}, 'entities.urls.0.display_url':/^instagram.com/}

    We first get the URLs, then get the top image from that URL using newspaper.
    then we use this to get the image. 
    '''
    import re

    # get all tweets with instagram url 
    regex = re.compile('^instagram.com')
    query = {'original':{'$exists':True}, 'img':{'$exists':False},\
    'img_watson':{'$exists':False},\
    'original.entities.urls.0.display_url':{'$regex':regex}}
    results = dbc.find(query, no_cursor_timeout=True)

    for doc in results:
        
        # returns: title, keywords, summary, top_img_src
        resolved_url = doc['original']['entities']['urls'][0]['expanded_url']
        
        try:
            url = summarise.get_top_img(resolved_url)
        except Exception as e:
            print('link summarise had error: ', e)

        # save the results back to that tweet
        if not url:
            continue

        else:
            # have the URL of img, so now classify it:
            data = summarise_algorithmia(url)
            if data:
                dbc.update({'_id':doc['_id']}, {'$set':data })
        
            data = summarise_watson(url)
            if data:
                dbc.update({'_id':doc['_id']}, {'$set':data })

    del results


def summarise_algorithmia(url, options={'threshold':0.15}, img_key='img'):
    data = {}
    try:
        output = summarise.summarise_img(url, options)
        summary = output.result
        if 'general' not in summary:
            print('no web service classified image data in response')
        else:
            data = {img_key + '.keywords': summary['general']}

    except Exception as e:
        print('algo summary failed:', e)

    return data


def summarise_watson(url, img_key='img_watson'):
    data = {}
    try:
        output = summarise.summarise_watson_img(url)
        summary = output['images'][0]
        # save the results back to that tweet
        if 'error' in summary:
            print('no web service classified image data in response')
        else:
            data = {img_key: summary['classifiers'][0]['classes']}

    except Exception as e:
        print('watson summary failed:', e)

    return data 


def summarise_images(dbc, options={'threshold':0.15}, watson=False ):
    '''
    Populate a mongo collection with image summaries for 
    its tweets' media_url image entities.
    '''
    count = 0 # keep track of how many records we update

    img_key = 'img'
    if watson:
        img_key = 'img_watson'

    # get all tweets with urls, but not had this img classing done before:
    query = {'original.entities.media.0.media_url':{'$exists':True},\
    img_key:{'$exists':False}}
    results = dbc.find(query, no_cursor_timeout=True)
    
    for doc in results:
        
        # returns: title, keywords, summary, top_img_src
        resolved_url = resolve_redirect(\
            doc['original']['entities']['media'][0]['media_url'])
        print('try url:', doc['original']['entities']['media'][0]['media_url'])

        if resolved_url == False:
            print('Url? ', doc['original']['entities']['media'][0]['media_url'])
            continue

        data = {}

        # which method to use:
        if not watson:
            try:
                output = summarise.summarise_img(resolved_url, options)
                summary = output.result

            except Exception as e:
                print('algo summary failed:', e)
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
                print('watson summary failed:', e)
                continue
        
            # save the results back to that tweet
            
            if 'error' in summary:
                print('no web service classified image data in response')
                continue
            else:
                data = {img_key: summary['classifiers'][0]['classes']}
        
        # if we made it this far, update the db record.
        dbc.update({'_id':doc['_id']}, {'$set':data })
        count += 1

    # clean up the cursor of pyMongo
    del results
    return count


def check_locations(dbc, region='UK'):
    '''
    batch search a collection for locations in its tweets'
    parsed text tokens
    '''
    query = {'txt':{'$exists':True}, 'has_location':{'$exists':False}}
    results = dbc.find(query, no_cursor_timeout=True)
    blacklist = []
    with open('uk_location_word_list.txt') as file:
        blacklist = file.read().splitlines()
    
    for doc in results:
        check_for_location(doc, region, blacklist)

        # todo: update a db field to show this has a location or not

    print('locations search complete')
    del results
        

def check_for_location(tweet, region, blacklist):
    '''
    Pass a tweet and check a lookup db for any strings that match a UK location

    # TODO: annotate tweets with features:
        # get a tweet

        # lookup a keyword in hashtags for a location
        # {'entities.hashtags.text':{$exists:true}} 
        + {'entities.hashtags.text':1, 'text':1} 
        # lookup a keyword in tweet (normalised) tokenised text for a location
    '''
    import loc_lookup
    # basic logging for this task.
    import logging
    FORMAT = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename="log_locs.txt", \
        level=logging.INFO, format=FORMAT)

    
    try:
        # we should search bigrams first as more reliable re: false positives
        bigrams = [' '.join(elm) for elm in tweet['txt']['bigrams']]
        result = loc_lookup.lookup(bigrams)[0]

        if not result:
            unigrams = tweet['txt']['parsed']
            unigrams_cleaned = \
            [token for token in unigrams if (token.lower()) not in blacklist]
            result = loc_lookup.lookup(unigrams_cleaned)
            
        # capture found locations so we can screen for false positives etc
        if len(result) > 1:
            found = ' '.join(result[1])
            if len(found) > 1:
                logging.info(str(tweet['_id']) + ': ' + found)

    except Exception as e:
        logging.error('lookup failed: ' + e)
        return 0

    return result[0]


if __name__ == '__main__':   
    '''
    Run various summaries to gather features and then do a classification on 
    twitter data.
    FIXME: should be improved with checking for duplicates 
    (i.e. of already accessed resources)
    '''
    ###########################################################################
    ## FEATURES gathering:
    # image summaries - using a web service
    ##

    coll = 'sample_a200'
    # summarise_instagram(jlpb.get_dbc('Twitter', coll))
    # print('done insta')
    # Uncomment below for URL extraction of summary on MongodDB
    # summarise_links(jlpb.get_dbc('Twitter', coll))
    check_locations(jlpb.get_dbc('Twitter', coll))

    exit('finsihed links summarising')
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

    ###########################################################################
    # CLASSIFYING:
    # params = [('txt','bigrams')]
    for param in params:

        ## TRAINING SET -------------
        train = []
        results = dbc.find()
        for doc in results:
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

        # wraps NLTK simply: return nltk.classify.accuracy(self.classifier, 
        # test_features) 
        acc = cl.accuracy(test) * 100
        print('Classifier Type      | ', type, ' with ', '.'.join(param))
        print('Accuracy, train/test | ', '=',  str(acc), '% ,', len(train), \
            '/', len(test))
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

    # get the label probability distribution with the prob_classify(text) method
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
    ## maybe try with one that removes key hashtag terms and see if it 
    # improves or not
    # A feature extractor is simply a function with document 
    # (the text to extract features from)
    # as the first argument.
    # The function may include a second argument, 
    # train_set (the training dataset), if necessary.
    #
    #


    # can try Noun Phrase extraction
