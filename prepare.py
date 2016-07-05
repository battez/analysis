'''
Process tweets; python 3.x 
'''
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
    '''
    Strip wide variety of URLs from a text
    '''
    link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)'\
        , re.DOTALL)
    links = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text


def strip_mentions(text):
    '''
    Remove @mentions i.e. usernames from tweet text
    '''
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
    '''
    Converts to lower case and cleans up the text.
    '''

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


def tokenise_tweet(tweet, stemmed=False, split=True):
    '''
    Pass this a pre-cleaned tweet string. We can then split it.
    stop words removed and remainder words stemmed if required
    '''
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
    '''
    Get some text features from a (normalised) tweet
    '''
    if split:
        tweet = tweet.split(' ')

    tweet_bigrams = bigrams(tweet)
    tweet_bigrams = list(tweet_bigrams)
    # for bigrammed in bigrams(tweet):
    #     tweet_bigrams['contains(%s)' % ','.join(bigrammed)] = True

    # list, dict as tuple
    return tweet_bigrams


def load_tokens(bigrams='data-bigram.csv', unigrams='data-unigram.csv'):
    '''
    Load in bigrams and unigrams from spreadsheet, that we will prune dataset by.
    '''
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

def retweet_stats(dbc):
    ''' 
    Return num. retweets and percentage
    '''
    total = dbc.count()
    r_total = dbc.count({'text':{'$regex':'^RT'}})
    percent = 100 * (r_total / total)
    return r_total, float("{0:.2f}".format(percent)) 


def reply_stats(dbc):
    ''' 
    Return num. replies and percentage
    '''
    total = dbc.count()
    r_total = dbc.count({'original.in_reply_to_tweet_id':{'$exists':1}, \
    'original.in_reply_to_tweet_id':{'$ne':None} })
    percent = 100 * (r_total / total)
    return r_total, float("{0:.2f}".format(percent))


def extract_tweet_entities(tweets):
    '''
    FIXME: extract common entities from tweets
    '''
    # ref: https://dev.twitter.com/docs/tweet-entities 

    if len(tweets) == 0:
        return [], [], [], [], []
    
    screen_names = [ user_mention['screen_name'] 
                         for tweet in tweets
                            for user_mention in tweet['entities']['user_mentions'] ]
    
    hashtags = [ hashtag['text'] 
                     for tweet in tweets 
                        for hashtag in tweet['entities']['hashtags'] ]

    urls = [ url['expanded_url'] 
                     for tweet in tweets 
                        for url in tweet['entities']['urls'] ]
    
    symbols = [ symbol['text']
                   for tweet in tweets
                       for symbol in tweet['entities']['symbols'] ]
    # In some circumstances (such as search results), the media entity
    # may not appear
    media = []
    for tweet in tweets:
       
        if 'media' in tweet['entities']: 
            
            media = media + [ media['display_url'] for media in tweet['entities']['media'] ]
        else:
            pass

    return screen_names, hashtags, urls, media, symbols


def get_common_entities(tweets, entity_threshold=3):
    '''
    Hashtags etc
    '''
    # Create a flat list of all tweet entities
    tweet_entities = [  entity
                        for tweet in tweets
                            for entity_type in extract_tweet_entities([tweet]) 
                                for entity in entity_type 
                     ]

    common = Counter(tweet_entities).most_common()

    # Compute frequencies
    return [ (key,val) for (key,val) in common if val >= entity_threshold ]

def summarise_entities(dbc, query=[{'$match':{'original':{'$exists':True}}} \
    , {'$project':{'entities':'$original.entities'}}], top=20):
    '''
    Display summary frequencies for entities in tweets; uses PrettyTable
    '''
    # Retrieve all the tweets from the database:
    # NB adjust query param if required for a standard set of tweets in a database
    tweets = dbc.aggregate(query)
    entities = []
    for tweet in tweets:
        entities.append(tweet)
    print('\nTotal No. Tweets retrieved: ' + str(len(entities)) )

    mentioned, hashtags, urls, media, symbols = extract_tweet_entities(entities)
    
    # normalise if needed:
    mentioned = [term.lower() for term in mentioned]
    hashtags = [term.lower() for term in hashtags]
    
    # freqs sets the entities to output:
    freqs={'@mentioned users':mentioned, 'hashtags':hashtags, 'media_urls':media}
    for kind, entity in freqs.items():
        count_all = Counter()
        count_all.update(entity)
        common = count_all.most_common(top)

        print('\nTotal No. ', kind + ': ', len(entity) )
        pt = PrettyTable(field_names=[kind, 'Count']) 
        [pt.add_row(kv) for kv in common]
        pt.align[kind], pt.align['Count'] = 'l', 'r' # Set column alignment

        # use a print wrapper to view this in case of strange non-unicode chars!
        jlpb.uprint(pt)
        del count_all        

# Both below: adapted from https://github.com/dandelany/tweetalyze/
def screen_names_in_db(dbc):
    '''
    Returns a list of all distinct Twitter screen names in the database.
    '''
    return dbc.distinct('original.user.screen_name') 


def total_tweets(dbc, threshold=1):
    '''
    Prints the total number of tweets for each screen name in the database.
    [['name', '# of tweets'], ['Dee_Marketing', 1], ['YounqFlexin_Dee', 1]]
    Can take a long time!
    '''
    export_data = [['name', '# of tweets']]
    
    for name in screen_names_in_db(dbc):
        
        query = dict({'original.user.screen_name': name}.items())
        amount = dbc.find(query).count()
        if amount > threshold:
            export_data.append([name, amount])

    return export_data

##
##
# Process some tweets
##
if __name__ == '__main__':
    '''
    clean up the text - normalise tweet-text content; 
    store tokenised and bigrams to file/db. 
    Also prune out any tweets that have invalid terms
    '''
    # for some output of results:
    from prettytable import PrettyTable


    # MongoDB data is from scraped tweets, so hashtag entities in original.entities
    dbc = jlpb.get_dbc('Twitter', 'rawtweets')
    
    # output some info on the entities:
    summarise_entities(dbc)
    exit()
    print('num. distinct users; (NB original only):', len(screen_names_in_db(dbc)))
    print(retweet_stats(dbc))
    print(reply_stats(dbc))

    
    exit()
    
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
        t_tweet = tokenise_tweet(n_tweet)

        phrases = tweet_features(t_tweet)
       
        # this tallies up bigrams and unigrams:
        count_all.update(phrases)
        count_all_uni.update(t_tweet)

                # count_all_hashtags.update()
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

    ##
    ##
    # Output some stats
    ##
    ##
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
    

    