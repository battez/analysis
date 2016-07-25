''' 
lookup locations UK open data. 
'''
import sys
CURR_PLATFORM = sys.platform
if CURR_PLATFORM == 'linux':
    exit('') # fixme:
else:
    sys.path.insert(0, 'U:\Documents\Project\demoapptwitter')

import config

# db connect:
import psycopg2
try:
    conn = psycopg2.connect(database="uk_places", user=config.DBPOSTGRES['user'],\
            password=config.DBPOSTGRES['password'], host="127.0.0.1", port="5432")
except psycopg2.OperationalError as e:
    print('Unable to connect!\n{0}').format(e)
    sys.exit(1)


cur = conn.cursor()




# NB category is 'type' in original csv.
schema_table = '''
    CREATE TABLE categories(
        id SERIAL PRIMARY KEY,
        name character varying(255)
);

'''
schema = '''
    CREATE TABLE locs(
        id SERIAL PRIMARY KEY,
        name1 character varying(255),
        category smallint,
        local_type smallint,
        district character varying(255),
        country smallint 
);
'''

'''
ALTER TABLE public.locs
ADD FOREIGN KEY (category) 
REFERENCES public.categories(id)
'''
# cur.execute(schema_table)
# conn.commit()
# exit()

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
            
            cur.execute('INSERT INTO locs (name1, category, local_type, district, country) VALUES (%s, %s, %s, %s, %s)'\
                , fields)

            conn.commit()
            
    return


def clean_field(field):
    '''
    Leave just letters numbers and spaces
    '''
    import re
    field = re.sub(r'[^a-zA-Z\d\s]','',field)
    return field.strip()


def load_csv(conn):
    '''
    Load in csv to database
    '''
    import os
    import csv
    # Load in the CSVs of lookup data
    # ensure we are in the right directory context, and then return to original at end. 
    previous_dir = os.getcwd()
    os.chdir(r'C:\Users\johnbarker\Downloads')

    directory = "opname_csv_gb"

    line_sep = "-" * 40 # 
    print(" ")
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

            
## uncomment line below to run populate load of CSV to db:
#load_csv(conn) # takes approx. 7 mins on i5 + SSD.

# tidy up db connection
conn.close()

#
# REMOTE connect
con = None
try:
    con = psycopg2.connect(database="uk_places", user=config.DBA['user'],\
            password=config.DBA['password'], host=config.DBA['host'], port="5432")
    remote_cur = con.cursor() 
    query = "select * from locs limit 5"
    remote_cur.execute(query)
    rows = remote_cur.fetchall()

    for row in rows:
        print (row)

except psycopg2.DatabaseError as e:
    print ('Error %s' % e )   
    sys.exit(1)

finally:
    
    if con:
        con.close()

''' # debug:
query = "SELECT * from LOCATIONS LIMIT 200"
cur.execute(query)
rows = cur.fetchall()


'''


