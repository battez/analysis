''' 
lookup locations UK open data. 

NB call populate_db() (see fun below) to run populate load of CSV to db.

This looks up candidate strings for location(s) in a postgres db of UK locations
and placenames. name1, district can be text searched. Type can also (see db)

'''
import sys
CURR_PLATFORM = sys.platform
if CURR_PLATFORM == 'linux':
    exit('') # fixme:
else:
    sys.path.insert(0, 'U:\Documents\Project\demoapptwitter')

import config

# Local Db connect:
import psycopg2


def process_row(row, conn=False):
    '''
    Process a row of a CSV, saving to database
    '''
    # fixme: load these categories from database ideally
    valid = ['populatedPlace', 'transportNetwork']
    local_types = sorted(['City', 'Hamlet', 'Village', 'Suburban Area', 'Town', 'Other Settlement', \
    'Named Road', 'Section Of Named Road', 'Section Of Numbered Road', 'Numbered Road'])
    countries = sorted(['Yorkshire and the Humber','London','North West','Wales','South West', \
    'Eastern','West Midlands','East Midlands','North East','South East','Scotland'])

    if row[6] in valid:
        
        # INSERT this location to db        
        if conn:
            cur = conn.cursor()
            
            values = [row[2], row[6], row[7], row[24], row[27]]
            values = list(map(clean_field, values))

            fields = (values[0], valid.index(values[1]) + 1, local_types.index(values[2]) + 1,\
             values[3], countries.index(values[4]) + 1) #tuple; add one for db index to match up
            
            cur.execute(\
            'INSERT INTO locs (name1, category, local_type, district, country) VALUES (%s, %s, %s, %s, %s)'\
                , fields)

            conn.commit()
            
    return


def clean_field(field):
    '''
    Leave only letters, numbers and spaces
    '''
    import re
    field = re.sub(r'[^a-zA-Z\d\s]','',field)
    return field.strip()


def load_csv(conn, directory):
    '''
    Load in a CSV file's data to database
    '''
    import os
    import csv

    # Load in the CSVs of lookup data
    # ensure we are in the right directory context, and then return to original at end. 
    previous_dir = os.getcwd()
    os.chdir(directory)

    directory = "opname_csv_gb"

    line_sep = "-" * 40 
    print(line_sep)
    print('PROCESSING, PLEASE WAIT....')
    print(line_sep)


    for root, dirs, files in os.walk(directory):
        
        for file in files:
            
            if file.endswith('.csv'):
                with open(directory + '\\' + file, 'r') as data:
                    
                    reader = csv.reader(data)

                    for row in reader:
                        process_row(row, conn)
                        

                data.close()

            

def populate_db(directory=None):
    
    if not directory:
        directory = r'C:\Users\johnbarker\Downloads'
    try:
        conn = psycopg2.connect(database="uk_places", user=config.DBPOSTGRES['user'],\
                password=config.DBPOSTGRES['password'], host="127.0.0.1", port="5432")
    except psycopg2.OperationalError as e:
        print('Unable to connect!\n{0}').format(e)
        sys.exit(1)


    cur = conn.cursor()

    load_csv(conn, directory) # takes approx. 7 mins on i5 + SSD.

    # tidy up db connection
    conn.close()


def lookup(values, district_search=True, limit=1, start_from=True, to_end=True):
    ''' 
    crude function to lookup two cols in a postgres db with regex
    of passed in list of strings
    '''
    # REMOTE connection
    conn = None
    try:
        conn = psycopg2.connect(database="uk_places", user=config.DBA['user'],\
                password=config.DBA['password'], host=config.DBA['host'], port="5432")
        cursor = conn.cursor() 
        
        # BUILD SELECT QUERY
        # Fixme: possibly speed improve the query with these syntax:
        # similar query:
        # select * from table where lower(value) similar to '%(foo|bar|baz)%';

        # regex -
        # select * from table where value ~* 'foo|bar|baz';
        query = "select name1, district from locs where locs.name1 ~* "
        alt_column = "locs.district"

        # precise regex matching or not:
        if start_from:
            query += "'^("

        else:
            query += "'("
            
        # regex case insensitive
        query += '|'.join(values)
        
        if to_end:
            query += ")$'"
            
        else:
            query += ")'"
            
        query += ' LIMIT ' + str(limit)
        
     
        ''' forms query like this:
        select name1,district from locs where 
        locs.name1 ~* '^(relief|government|little|set|billions|flood|relief|use|money)$' 
        LIMIT 1
        '''
        cursor.execute(query)
        rows = cursor.fetchall()
        
        location = ''
        
        if cursor.rowcount > 0:
            # we got a match so lets return the call
            location = rows[0] # debugging

            return cursor.rowcount, location
        else:
            # check other table column
            cursor.execute(query.replace('locs.name1', alt_column))
            rows = cursor.fetchall()
            
            if cursor.rowcount > 0:
                location = rows[0] # debugging 
            
            return cursor.rowcount, location


    except psycopg2.DatabaseError as e:
        print ('Error %s' % e )   
        sys.exit(1)

    finally:
        
        if conn:
            conn.close()


if __name__ == '__main__':
    
    ''' 
    NB call populate_db() (see func above) to run populate load of CSV to db.

    This looks up candidate strings for location(s) in a postgres db of UK locations
    and placenames. name1, district can be text searched. Type can also (see db)
    '''

    candidates = ['Burragarth', 'Birnam Crescent', 'WiDnes','taunton','Chess','wibble','nuneaton','cheshire','lanark','East kilbride'\
    ,'South Lanarkshire']
    test = ['Thunder', 'UKStorm','Ham', 'London'] # London issue!
    test2 = {
    "parsed" : [
            "flood", 
            "alert", 
            "test", 
            "upper", 
            "dee", 
            "valley", 
            "llanuwchllyn", 
            "llangolle", 
            "pic", 
            "twitter", 
            "com", 
            "ccrptjhsjw"
        ], 
        "bigrams" : [
          
            [
                "test", 
                "upper"
            ], 
            [
                "upper", 
                "dee"
            ], 
            [
                "dee", 
                "valley"
            ], 
            [
                "valley", 
                "llanuwchllyn"
            ], 
            [
                "llanuwchllyn", 
                "llangolle"
            ], 
            [
                "llangolle", 
                "pic"
            ],          
            [
                "dee", 
                "valley"
            ], 
            [
                "valley", 
                "llanuwchllyn"
            ], 
            [
                "llanuwchllyn", 
                "llangolle"
            ], 
            [
                "llangolle", 
                "pic"
            ]
        ] 
    }

    bigrams = [' '.join(elm) for elm in test2['bigrams']]
    t = ['Sxt Vincent Street']
    match = lookup(t , limit=1)
    print('match: ', match[0])

