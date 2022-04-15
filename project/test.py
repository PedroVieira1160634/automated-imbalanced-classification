print("inicio")

import sys
#import csv

reading_file = open(sys.path[0] + '/output/results.csv', "r")

new_file_content = ""
i=0
j=7-1

for line in reading_file:
    stripped_line = line.strip()
    
    if j <= i < (j+5):
        new_line = stripped_line.replace(stripped_line, "new string")
    else:
        new_line = stripped_line
        
    new_file_content += new_line +"\n"
    i+=1
reading_file.close()

writing_file = open(sys.path[0] + '/output/results.csv', "w")
writing_file.write(new_file_content)
writing_file.close()


    
    
print("fim")