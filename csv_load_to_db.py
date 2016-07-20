# batch_clean.py
# by JLP Barker
# AC52048 Business Intelligence
# March 2016. 
# Process web server log files for easy input to SSIS .
# Usage: ensure folder for processing specified
#
# 
# Require some standard libraries:
import os

# ensure we are in the right directory context, and then return to original at end. 
previous_dir = os.getcwd()
os.chdir('U:\Documents\Bus Int')

directory = "W3SVC1"
field_string = "#Fields"

line_sep = "-" * 40
print(" ")
print(line_sep)
print('PROCESSING, PLEASE WAIT....')
print(line_sep)


field_line_types = []

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.log'):
            with open(directory + '\\' + file) as text:
                
                for line_num, line in enumerate(text):
                    if(line[0:7].lower() == field_string.lower()):
                        field_line_types.append(line)
            text.close()

# find the unique field-lengths:
set_field_line_types = set(field_line_types)

# diagnose 
# print(set_field_line_types)

# get the number of fields in each type and keep the headers for use in output
num_fields = []
headers = []
orig_headers = dict()
for kind in set_field_line_types:
    pieces = kind.split()

    # store this header string for use when we output later on (below)
    orig_headers[str(len(pieces) - 1)] = kind

    # remove the first field as we dont count #fields, of course!
    pieces.pop(0)
    num_fields.append(len(pieces))
    headers.append(pieces)


# initialise output files, with fields as headers:
FPREFIX = 'serverlog'
EXT = '.log'
OUTDIR = 'flatfiles\\' 

for idx, header in enumerate(headers):
    
    # open a new file
    with open(OUTDIR + FPREFIX + str(len(header)) + EXT, "wt") as file:
        # join and output to file
        header_string = ' '.join(header)
        file.write(header_string + "\n")


    file.close()


# Reopen our output files in append mode, before we start finding the data for them:
outfile14 = open(OUTDIR + FPREFIX + str(14) + EXT, "a")
outfile18 = open(OUTDIR + FPREFIX + str(18) + EXT, "a")  

# loop through the log files and write out no comment lines as data
# to relevant output file (ie with same no. of fields in its header, 
# as the data)
total_rows = 0
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.log'):
            with open(directory + '\\' + file) as text:

                # initialise this var to tell us when to save a line to output:
                capture_data = False

                for line in text:

                    # turn on data capture whenever we find a header line
                    if line[0:7].lower() == field_string.lower():
                        capture_data = True

                        # set which output file we will write to:
                        # if line == orig_headers['14']:
                        
                        #     outfile = outfile14
                        # elif len(line.split()) == 18:
                        #     outfile = outfile18
                        

                    elif line[0] == "#":
                        # just discard comments, reset capturing of data
                        capture_data = False

                    elif (line[0:2] == "20") and capture_data:

                        # run a check for field number in line:
                        # set which output file we will write to:
                        if len(line.split()) == 14:
                            outfile = outfile14
                            total_rows +=1
                            outfile.write(line)

                        elif len(line.split()) == 18:
                            outfile = outfile18
                            total_rows +=1
                            outfile.write(line)
                        else:
                            continue # discard line
                        
                        

                text.close

outfile14.close()
outfile18.close()
print( " " )
print(line_sep )
print('processing complete!')
print(line_sep)
print( " " )
print(line_sep)
print 'total rows=',total_rows
print(line_sep)
print(line_sep)

os.chdir(previous_dir)
                



           


