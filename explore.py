'''
Explore the flood stream set.
'''
# key stem word of flood in mongoDB: 1060 tweets from the
# streamed data set. 
# In this period a lot of flash flooding, especially in the populous S. England
# including floods in London and around on the Day of the EU referendum.
# date ranges from 
# 18h 20.06.2016 -- 17h 29.06.16
# 1060 tweets
# 990 classed as relevant by manual labelling. 
# (i.e. remaining 70 were other senses of the word flood and so on).
# 528/990 relevant have coordinates geospatial location.
# 
# TODO: Compare with Random sample of this set! 990? OR irrelevant labelled? 
# Compare with winter relevant dataset also. 
# Ask Andy if data analysis ok for thesis?
# 
# for some output of results:

from collections import Counter

import prepare as prp
import jlpb


def print_common(freqs, num=None, print_to_file=False, suffix=''):
    '''
    nicely print out some ngram frequencies, 
    handling any strange unicode data output.

    Takes a dict with the column heading as key,
    the data should be a colelctions Counter object.
    '''
    from prettytable import PrettyTable
    from jlpb import uprint # unicode printing

    if print_to_file:
        
        import uuid
        import csv
        headers = ['Words ' + suffix, 'Word Count']

        for key, counted in freqs.items():

            # open file to write to
            filename = key + '_' + str(uuid.uuid4()) + '.csv'
            output_file = open(filename, 'w', newline='')
            writer = csv.writer(output_file)
            writer.writerow(headers)  
            common = counted.most_common(num)
            for row in common:
                
                if type(row[0]) is not tuple:
                    writer.writerow(list(row))
                else: 
                    writer.writerow([' '.join(row[0]), row[1] ])    
                
            output_file.close()
            

    else: 
        # print to display
        for key, counted in freqs.items():
            
            common = counted.most_common(num)
            
            pt = PrettyTable(field_names=[key, 'Count']) 
            [pt.add_row(kval) for kval in common]
            pt.align[key], pt.align['Count'] = 'l', 'r' # Set column alignment

            # use a print wrapper to view this in case of strange non-unicode!
            uprint(pt)



if __name__ == '__main__':
    #
    # one-off process Report CSV data
    #
    #
    dbc = jlpb.get_dbc('Twitter', 'reports')    
    import re
    # initialise counters 
    count_all = Counter()
    count_all_uni = Counter()
    count_all_tri = Counter()
    num = None # how many of top word-counts, descending, to show

    results = dbc.find({}).sort([("FLOOD_CATEGORY_DESC", 1)])
    current_category = ''
    for doc in results[:]:
        
        txt = doc['FLOOD_DESC']
        
        # do this separately for each category
        if doc['FLOOD_CATEGORY_DESC'] != current_category:
            
            # output the most recent category first
            if current_category != '':
                print('printing', current_category)
                print_common({'Unigram': count_all_uni, 'Bigram':count_all,\
                    'Trigram':count_all_tri}, \
                    None, True, "_".join(re.findall("[a-zA-Z]+",\
                        current_category)) )
            
            # assign the new category to current
            current_category = doc['FLOOD_CATEGORY_DESC']

            # re-initialise the counters
            count_all = Counter()
            count_all_uni = Counter()
            count_all_tri = Counter()

        # tokenise into various N-grams:
        n_tweet = prp.normalise_tweet(txt)
        t_tweet = prp.tokenise_tweet(n_tweet)
        phrases = prp.tweet_features(t_tweet)
        tri_grams = prp.tweet_trigrams(t_tweet)

        # this tallies up various counts for these unigrams, bigrams etc:
        count_all.update(phrases)
        count_all_tri.update(tri_grams)
        count_all_uni.update(t_tweet) 

        '''
        dbc.update({'_id':doc['_id']}, {\
        # '$push':{'desc_trigrams': {'$each':tri_grams},\
        # 'desc_bigrams': {'$each':phrases}},\
        '$set':{'desc_unigrams':' '.join(set(t_tweet))}\
        })
        '''
    
    # print out last category:
    print_common({'Unigram': count_all_uni, 'Bigram':count_all,\
        'Trigram':count_all_tri}, \
     None, True, "_".join(re.findall("[a-zA-Z]+", current_category)) )


    exit('CSV data only run. Remove me to use MongoDb')



    ##
    #
    # end analyse the CSV data.
    ##

    # MongoDB data is from scraped tweets,
    # so hashtag entities in original.entities
    if prp.CURR_PLATFORM != 'linux':
        #stream2flood_all, sample_stream2_990,sample_stream2_brexit,
        #sample_stream2_rain,stream2_storm_all
        dbc = jlpb.get_dbc('Twitter', 'stream2flood_all') 
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

    