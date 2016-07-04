''' process tweets; python 2.x '''
import sys
import pymongo
TWITDIR = 'U:\Documents\Project\scrape'
sys.path.insert(0, TWITDIR)

# get some handy functions 
import jlpb
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from nltk import bigrams

import re, string, json
from pprint import pprint

from collections import Counter


def strip_links(text):
    '''strip wide variety of URLs from a text'''
    link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)'\
        , re.DOTALL)
    links = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text


def strip_mentions(text):
    '''remove @mentions i.e. usernames from tweet text'''
    entity_prefixes = ['@'] # can use for hastags too, if needed
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)


def normalise_tweet(tweet, nums=True):
    '''converts to lower case and cleans up the text.'''

    # Various regular expressions used to clean up the tweet data
    # remove_ellipsis_re = re.compile(r'[\.{2}\u2026]')
    # \.{2,}(.+)
    remove_ellipsis_re = re.compile(r'[^\.]\.{2,3}')
    punct_re = re.compile(r"[\"'\[\],’#.:;()&!\u2026]") # leave hyphens
    number_re = re.compile(r"\d+")

    # lowercase all
    tweet = tweet.lower()
    # remove apostrophes
    tweet = jlpb.strtr(tweet, {"'":''})

    # remove links
    tweet = strip_links(tweet)
    # remove usernames
    tweet = strip_mentions(tweet)
    # remove ellipses
    tweet = re.sub(remove_ellipsis_re, '', tweet)
    # remove various punctuation
    tweet = re.sub(punct_re, '', tweet)
    
    # remove numbers:
    if nums:
        tweet = re.sub(number_re, '', tweet)

    return tweet


def tokenise(tweet, stemmed=False, split=True):
    '''Pass this a pre-cleaned tweet string. We can then split it.
    stop words removed and remainder words stemmed if required'''
    #Remove the stop words.
    if split:
        tweet = tweet.strip().split()
    more_stopwords = stopwords.words('english') + ['u', 'ur', 'yr', 'k']

    tweet_parsed = [word for word in tweet if word not in more_stopwords]

    #Lemmatize or stem the words.
    if stemmed:
        stemmer = LancasterStemmer()
        tweet_parsed = [stemmer.stem(word) for word in tweet_parsed]

    return tweet_parsed


def tweet_features(tweet, split=False):
    '''get some text features from a (normalised) tweet'''
    if split:
        tweet = tweet.split(' ')

    tweet_bigrams = bigrams(tweet)
    tweet_bigrams = list(tweet_bigrams)
    # for bigrammed in bigrams(tweet):
    #     tweet_bigrams['contains(%s)' % ','.join(bigrammed)] = True

    # list, dict as tuple
    return tweet_bigrams


if __name__ == '__main__':
    ''' clean up the text - normalise tweet-text content; store to file/db'''
    # MongoDB data is from scraped tweets, so hashtag entities in original.entities
    dbc = jlpb.get_dbc('Twitter', 'rawtweets')
    dbstore = jlpb.get_dbc('Twitter', 'rawtweets_prepared')
    
    # for some output of results:
    from prettytable import PrettyTable
    total_num = dbc.count()

    # get the scraped tweets from mongodb, possibly only use English (?), 
    # that we could supplement from the API:
    results = dbc.find() #
    print('Total tweets gathered from queries:', total_num)

    total_num_orig = dbc.find({"original":{'$exists':True}}).count()
    print('Total tweets from queries, supplemented by Twitter API info:', total_num_orig)

    count_all = Counter()
    invalidate_phrases = [('taylor','reynolds'),('jack','shinnie'),('logan','shinnie'),\
    ('buttahatchie', 'river'),('warning', 'buttahatchie'),('pee','dee'),('hamstring','injury'),\
    ('considine', 'taylor'),('willo','flood'),('warning','pee'),('little','pee'),('river','pee'),\
    ('midfielder','willo'),('willo','floods'),('reynolds','considine'),('anderson','reynolds')]

    # single terms we can consider as meaning the tweet is irrelevant
    invalidate_terms = ['longoria', 'considine', 'shinnie', 'hoquiam']
    
    # update this batch in the database:
    for doc in results[:]:
        # use the original text as twitter provides this in most suitable format 
        # (as compared to the rendered text of the scraped tweet)
        if 'original' in doc:
            txt = doc['original']['text']
        else:
            txt = doc['text'] # fall back to this otherwise
        
        n_tweet = normalise_tweet(txt)
        t_tweet = tokenise(n_tweet)

        phrases = tweet_features(t_tweet)
       
        # this tallies up bigrams:
        count_all.update(phrases)

        # we check that this tweets bigrams are not in the list of bad bigrams:
        invalidate = False
        for invalid in invalidate_phrases:

            # we will delete it from the db:
            if invalid in phrases:
                invalidate = True
                break
        
        for invalid in invalidate_terms:

            # we will delete it from the db:
            if invalid in t_tweet:
                invalidate = True
                break

        if invalidate:
            # delete  this document
            dbc.remove({'_id':doc['_id']})
            continue
            
        else:
            # comment out this line to run the updates below if necessary: 
            continue

            # insert as a nested field of the raw tweet we already have for this ID
            dbc.update({'_id':doc['_id']}, {\
                '$push':{'txt.bigrams': {'$each':phrases}},\
                '$set':{'txt.normalised':n_tweet,'txt.parsed':t_tweet}\
                })

    # view our most frequent bigrams    
    common = count_all.most_common(5)

    pt = PrettyTable(field_names=['Bigram', 'Count']) 
    [pt.add_row(kv) for kv in common]
    pt.align['Bigram'], pt.align['Count'] = 'l', 'r' # Set column alignment

    # use a print wrapper to view this in case of strange non-unicode chars!
    jlpb.uprint(pt)
    
    


    