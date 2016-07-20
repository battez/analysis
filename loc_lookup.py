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
# FIXME: relational with eg : ,
#       local_type INT NOT NULL references author(id) 
# NB category is 'type' in original csv.
schema = '''
    CREATE TABLE locations(
        id SERIAL PRIMARY KEY,
        name1 character varying(255),
        category character varying(255),
        local_type character varying(255),
        district character varying(255),
        country character varying(255)
);

'''
# cur.execute(schema)
# conn.commit()
# exit()


def process_row(row, conn=False):
    '''
    Process a row of a CSV, saving to database
    '''
    valid = ['populatedPlace', 'transportNetwork']

    """ use if need to filter out:
    local_type = ['City', 'Hamlet', 'Village', 'Suburban Area', 'Town', 'Other Settlement', \
    'Named Road', 'Section Of Named Road', 'Section Of Numbered Road', 'Numbered Road']
    """ 
    if row[6] in valid:
        
        print(clean_field(row[2]), row[6], 'keep',row[7], row[24], row[27]) 
        # if row[7] not in local_type:
        #     exit(row[7])
        if conn:
            cur = conn.cursor()
            query = '''
            INSERT INTO locations (name1, category, local_type, district, country)
            VALUES (
            '''
            values = [row[2], row[6], row[7], row[24], row[27]]
            
            values = list(map(clean_field, values))
            
            table_name = 'locations'
            
            # FIXME :cur.execute("INSERT INTO  VALUES ('{0}')", format(values)) 
            query = "INSERT INTO %s (name1, category, local_type, district, country) VALUES (%%s, ...)" % table_name
            cur.execute(query, values)

            conn.commit()
            exit()
    return



def clean_field(field):
    '''
    Leave just letters numbers and spaces
    '''
    import re
    field = re.sub(r'[^a-zA-Z\d\s]','',field)
    return field


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

            
                

load_csv(conn)





query = "SELECT * from LOCATIONS"
cur.execute(query)


rows = cur.fetchall()

count = 0
for row in rows:
    if count > 4:
        break

    print ("ID = ", row[0])
    print ("name1 = ", row[1])
    print ("TYPE = ", row[2])
    print ("localTYPE = ", row[3])
    print ("district = ", row[4])
    print ("country = ", row[5], "\n")
    count += 1


# tidy up db connection
conn.close()
