'''
Explore the flood stream set.
'''
# key stem word of flood in mongoDB: 1060 tweets from the
# streamed data set. 
# In this period a lot of flash flooding, especially in the populous south of England
# including floods in London and around on the Day of the EU referendum.
# date ranges from 
# 18h 20.06.2016 -- 17h 29.06.16
# 1060 tweets
# 990 classed as relevant by manual labelling. 
# (i.e. remaining 70 were other senses of the word flood and so on).
# 528/990 relevant have coordinates geospatial location.
# 
# TODO: Compare with Random sample of this set! 990? OR irrelevant labelled? 
# Compare with winter relevant dataset also. Ask Andy if data analysis ok for thesis?
# 
# for some output of results:
from prettytable import PrettyTable
from collections import Counter

import prepare as prp
import jlpb


if __name__ == '__main__':
    

    # MongoDB data is from scraped tweets, so hashtag entities in original.entities
    if prp.CURR_PLATFORM != 'linux':
        dbc = jlpb.get_dbc('Twitter', 'stream2_storm_all') #stream2flood_all
    else:
        dbc = jlpb.get_dbc('local', 'stream2flood_all')

    query = {'t_class':{'$eq':1}}
    query = {}
    prp.summarise_entities(dbc, query=[{'$match':query}])

    exit()
    # counters for freqDists
    count_all = Counter()
    count_all_uni = Counter()
    count_all_tri = Counter()
    num = 10 # how many to show

    results = dbc.find(query).sort([("date_created_at", 1)])


    for doc in results[420:440]:
        txt = doc['text']
        #print(doc['created_at'])

        n_tweet = prp.normalise_tweet(txt)
        jlpb.uprint(n_tweet)   
        t_tweet = prp.tokenise_tweet(n_tweet)

        # bigrams etc:
        phrases = prp.tweet_features(t_tweet)
        tri_grams = prp.tweet_trigrams(t_tweet)

        # this tallies up bigrams and unigrams:
        count_all.update(phrases)
        count_all_tri.update(tri_grams)
        count_all_uni.update(t_tweet) 

    exit('ends -----------')
    