'''
summarise url (s) and other web resources
'''
from newspaper import Article, Config


def summarise_one(url, title=True, keywords=True, summary=False, top_img_src=False):
    '''
    Get url and return summary 
    '''
    article = Article(url)
    # configuration for Newspaper to minimize processing time
    configure = Config()
    configure.fetch_images = False
    configure.MAX_SUMMARY = 300
    configure.MAX_SUMMARY_SENT = 3
    
    try:
        article.download()
        article.parse()
    except:
        print(url) 

    
    title = article.title
    if keywords or summary:
        try:
            article.nlp()
            if keywords:
                keywords = article.keywords
            if summary:
                summary = article.summary
        except :
            print('NEwspaper error with nlp() call')
        
    if top_img_src:
        top_img_src = article.top_image
   
    return title, keywords, summary, top_img_src


def summarise_img(src, options=False):
    '''
    Retrieve meta-data for an image web resource.
    Use algorithmia, openshift or similar cloud service. 
    '''
    import sys
    CURR_PLATFORM = sys.platform
    if CURR_PLATFORM == 'linux':
        return None
    else:
        sys.path.insert(0, 'U:\Documents\Project\demoapptwitter')
    import config

    import Algorithmia
    client = Algorithmia.client(config.ALGORITHMIA['api_key'])
    algo = client.algo('deeplearning/IllustrationTagger/0.2.3')

    input = {"image":src}
    if options:
        # tags (optional) required probs
        for opt, value in options.items():
            input[opt] = value

        # e.g. threshold  0.3 etc
        

    result = algo.pipe(input)

    '''
    Returns something like:

{'copyright': [{'real life': 0.5092955231666565}], 'character': [], 'general': [{'photo': 0.9757122993469236}, {'1girl': 0.788438677
7877808}, {'solo': 0.6956651806831361}, {'lips': 0.35438942909240734}, {'face': 0.27816683053970337}], 'rating': [{'safe': 0.9757491
946220398}, {'questionable': 0.02385559491813184}, {'explicit': 0.0013747639022767544}]}

{'character': [], 'general': [{'no humans': 0.5748320221900941}, {'solo': 0.2691210508346558}, {'sky': 0.21563498675823217}, {'1girl
': 0.19651243090629583}, {'cloud': 0.17825683951377871}, {'tree': 0.14194014668464663}, {'window': 0.1255052238702774}], 'rating': [
    '''
    return result
    

# for testing use:
if __name__ == '__main__':    
    url = input('Enter URL: ')
    if(url[-3:]) not in ['jpg','png','gif']:
        output = summarise_one(url, False, False, True, True)
        title, keywords, summary, top_img_src = output
        print('title', title)
        print('keywords', keywords)
        if type(summary) is not bool:
            print('summary', summary.encode('utf-8'))
        print('top_img_src', top_img_src)
    else:
        #'threshold':0.1,
        # options = {'tags':[]}
        # options = { 'tags':['tree','window','face','lips', 'sky']}
        output = summarise_img(url)
        print(output.result)

    

