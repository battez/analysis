''' process tweets; python 2.x '''
import sys
import pymongo
TWITDIR = 'U:\Documents\Project\scrape'
sys.path.insert(0, TWITDIR)
import jlpb
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
# get some handy functions 


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

    # regular expressions used to clean up the tweet data
    # remove_ellipsis_re = re.compile(r'[\.{2}\u2026]')
    # \.{2,}(.+)
    remove_ellipsis_re = re.compile(r'[^\.]\.{2,3}')
    
    punct_re = re.compile(r"[\"'\[\],’#.:;()&!\u2026]") # leave hyphens
    number_re = re.compile(r"\d+")

    tweet = tweet.lower()
    
    # remove links
    tweet = strip_links(tweet)
    
    # remove usernames
    tweet = strip_mentions(tweet)

    tweet = re.sub(remove_ellipsis_re, '', tweet)
    
    tweet = re.sub(punct_re, '', tweet)
    
    # uncomment, to remove numbers:
    if nums:
        tweet = re.sub(number_re, '', tweet)

    
    return tweet


def parse_tweet(tweet, stemmed=False, split=True):
    '''Pass this a pre-cleaned tweet string. We can then split it.
    stop words removed and remainder words stemmed if required'''
    #Remove the stop words.
    if split:
        tweet = tweet.strip().split(' ')
    more_stopwords = stopwords.words('english') + ['u', 'ur', 'yr', 'k']

    tweet_parsed = [word for word in tweet if word not in \
    more_stopwords]

    #Lemmatize or stem the words.
    if stemmed:
        stemmer = LancasterStemmer()
        tweet_parsed = [stemmer.stem(word) for word in tweet_parsed]

    return tweet_parsed


def tweet_features(tweet):
    '''get some text features from a tweet'''
    tweet_unigrams = tweet.split(' ')
    tweet_bigrams = {}
    for bigrams in nltk.bigrams(tweet_unigrams):
        tweet_bigrams['contains(%s)' % ','.join(bigrams)] = True

    # list, dict as tuple
    return tweet_unigrams, tweet_bigrams


if __name__ == '__main__':
    ''' clean up the text - normalise tweet-text content; store to file/db'''
    T4 = '''Be one of the first to join to take advantage of the flood of traffic your ads will receive every day.
You Need Leads and You Need Sign-Ups'''
    T3 = 'Tremendous save from the County keeper to dent Flood but I think offside might have been given anyway. http://bytheminsport.com/events/1772-football-aberdeen-ross-county-follow-the-final-match-of-the-season-from-pittodrie-with-us-minute-by-minute-live-15-may-2016 …'
    T5 = '''@kanami_nico どうもですｗ'''
    TEST_T = 'Sorry for the flood but I had to let it out I have nobody to vent to so ...'
    T2 = '15K as relief from the government is so little. They set up billions for flood relief, use that MONEY'
    # MongoDB data is from scraped tweets, so hashtag entities in original.entities
    dbc = jlpb.get_dbc('Twitter', 'rawtweets')
    dbstore = jlpb.get_dbc('Twitter', 'rawtweets_prepared')

    
    # FIXME: check for language in tweet JSON; original.lang == 'en' 
    try:
        # debug
        pass
        #tweet = normalise_tweet(T5)
        # print(tweet)
        #tweet = parse_tweet(tweet)
        # print('parsed', tweet)
    
    except Exception as e:
        print('error')
        raise e
    
    
    # get the tweets from mongodb:
    results = dbc.find({"original":{'$exists':True}})

    # update this batch in the database:
    for doc in results:
        txt = doc['original']['text']
        print(doc['_id'])
        n_tw = normalise_tweet(txt)
        p_tw = parse_tweet(n_tw)

        # TODO: add tweet_features also
        # insert as a nested field of the raw tweet we already have for this ID
        dbc.update({'_id':doc['_id']}, {\
            '$set':{'txt.normalised':n_tw,'txt.parsed':p_tw}\
            })
        

    del results


    