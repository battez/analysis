'''
load tweets in one from a mongo DB and present them to a user.
Who can label them with a binary class, and then update the database.

'''
# we use random
from random import shuffle
import os, sys
import pymongo

# set needed filepaths depending on OS 
CURR_PLATFORM = sys.platform
MACDIR = '~/Dropbox/data-notes-mac-to-chrome/data-incubator/Project_submission/supporting_files_code_queries_logs_Etc/'

if CURR_PLATFORM == 'darwin':
    TWITDIR = os.path.expanduser(MACDIR + 'demoapptwitter')
    SCRAPEDIR = os.path.expanduser(MACDIR + 'scrape')

else:
    TWITDIR = '/home/luke/programming/'
    SCRAPEDIR = '/home/luke/programming/scraping'

sys.path.insert(0, TWITDIR)
sys.path.insert(0, SCRAPEDIR)

# get some handy functions 
import jlpb
import jlpb_classify # prepare too?

def quick_label_tweets(dbt, dborig, skip=0):
    '''
    run through a mongodb collection of tweets, prompting -
    asking user to label the tweet before moving to next one.
    '''
    docs = dbt.find()[skip:]
    count = 0
    response = ''

    for doc in docs:

        place = doc['place']['full_name']
        
        stop = False
        count += 1

        while True:
            response = input( jlpb.uprint(doc['text'] + \
                        "\n user:" + doc['user']['screen_name'] + \
                            "\n place:" + place + \
                            "\nhttp://www.twitter.com/statuses/"+doc['id_str']) )
            if response == 'x':
                stop = True
                break
            elif response == 'y':
                # update db record
                dborig.update_one({
                  'id': doc['id']
                },{
                  '$set': {
                    't_class': 1
                  }
                }, upsert=False)
                response = 'yes'
                
            else:
                dborig.update_one({
                  'id': doc['id']
                },{
                  '$set': {
                    't_class': 0
                  }
                }, upsert=False)
                response = 'n'

            print('...for ' + str(count) + ', you said ', response.upper())
            print('------------------------------------')
            break

        if stop:
            break

    print('docs complete: ', count)

if __name__ == "__main__":

    # uncomment to label:
    # db & collection of tweets to classify
    # dbt = jlpb.get_dbc('Twitter', 'labelled')
    # dborig = jlpb.get_dbc('Twitter', 'risk_tweets')
    # quick_label_tweets(dbt, dborig, 2462)


