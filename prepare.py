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

def load_tokens(bigrams='data-bigram.csv', unigrams='data-unigram.csv'):

    # Load in bigrams and unigrams from spreadsheet, that we will prune dataset by.
    from csv import reader
    invalidate_phrases = list()
    invalidate_terms = list()

    if len(bigrams):
        # two column CSV of terms
        with open(bigrams) as f:
            invalidate_phrases = [tuple(line) for line in reader(f)] 

    if len(unigrams):
        # single column CSV of terms
        with open(unigrams) as f:
            invalidate_terms = [line[0] for line in reader(f)] 

    return invalidate_phrases, invalidate_terms

if __name__ == '__main__':
    ''' clean up the text - normalise tweet-text content; 
    store tokenised and bigrams to file/db. 
    Also prune out any tweets that have invalid terms'''
    
    # MongoDB data is from scraped tweets, so hashtag entities in original.entities
    dbc = jlpb.get_dbc('Twitter', 'rawtweets')
    
    # for some output of results:
    from prettytable import PrettyTable
    total_num = dbc.count()
    # store a frequency tabulation using Counter()s:
    count_all = Counter()
    count_all_uni = Counter()
    num = 10 # how many to show
    
    # Get the scraped tweets from mongodb, possibly only use English (?), 
    # that we could supplement from the API:
    results = dbc.find() # dbc.find({"original":{'$exists':True}}).count()

    # load in from CSV n-grams we will spot and then delete corresponding rows in database:
    invalidate_phrases, invalidate_terms = load_tokens()

    # delete invalidated tweets and then update tweets in the database, with parsed text
    for doc in results[:]:
        
        if 'original' in doc:
            # use the original text as twitter provides this in most suitable format 
            # (as compared to the rendered text of the scraped tweet)
            txt = doc['original']['text']
        else:
            txt = doc['text'] # fall back to this otherwise
        
        n_tweet = normalise_tweet(txt)
        t_tweet = tokenise(n_tweet)

        phrases = tweet_features(t_tweet)
       
        # this tallies up bigrams:
        count_all.update(phrases)
        count_all_uni.update(t_tweet)
        # we check that this tweets bigrams are not in the list of bad bigrams:
        invalidate = False
        for invalid in invalidate_phrases:

            # delete it from the db:
            if invalid in phrases:
                invalidate = True
                break
        
        for invalid in invalidate_terms:

            # delete it from the db:
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
    common = count_all.most_common(num)

    pt = PrettyTable(field_names=['Bigram', 'Count']) 
    [pt.add_row(kv) for kv in common]
    pt.align['Bigram'], pt.align['Count'] = 'l', 'r' # Set column alignment

    # use a print wrapper to view this in case of strange non-unicode chars!
    jlpb.uprint(pt)

    common = count_all_uni.most_common(num)
    pt = PrettyTable(field_names=['Unigram', 'Count']) 
    [pt.add_row(kv) for kv in common]
    pt.align['Unigram'], pt.align['Count'] = 'l', 'r' # Set column alignment

    # use a print wrapper to view this in case of strange non-unicode chars!
    jlpb.uprint(pt)
    
    


    