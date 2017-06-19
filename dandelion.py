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

import requests
import json
import config

ENTITY_URL = 'https://api.dandelion.eu/datatxt/nex/v1'

def get_entities(text, confidence=0.1, lang='en'):
    payload = {
        '$app_id': config.DANDELION_APP_ID,
        '$app_key': config.DANDELION_APP_KEY,
        'text': text,
        'confidence': confidence,
        'lang': lang,
        'social.hashtag': True,
        'social.mention': True
    }
    response = requests.get(ENTITY_URL, params=payload)
    return response.json()
 
def print_entities(data):
    for annotation in data['annotations']:
        print("Entity found: %s" % annotation['spot'])
 
if __name__ == '__main__':
    # sample works:
    # query = "Heavy rainfall in Lincolnshire at the moment. It's heading east, so a chance later to head out with your cameras! https://t.co/XO64t99gBO"
    # returns no entity: 
    #query = '@aviationguycw Hello Callum. Sincere apologies that both vehicles were15 minutes late arriving at the school yesterday afternoon. Heavy rain'
    
    query = '@necky_fade I see other good players really struggled, lots of WD was there heavy rain like at Essendon?'
    response = get_entities(query)
    print(json.dumps(response, indent=4))

