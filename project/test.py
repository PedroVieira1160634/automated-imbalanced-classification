print("inicio")




#new write

#import sys
##import csv
#
#reading_file = open(sys.path[0] + '/output/results.csv', "r")
#
#new_file_content = ""
#i = 0
#j = 5           # 4 metrics + 1 seperator
#interval = 7-1  # if line to start is 7
#
#for line in reading_file:
#    stripped_line = line.strip()
#    
#    if interval <= i < (interval + j):
#        new_line = stripped_line.replace(stripped_line, "new string")
#    else:
#        new_line = stripped_line
#        
#    new_file_content += new_line +"\n"
#    i+=1
#reading_file.close()
#
#writing_file = open(sys.path[0] + '/output/results.csv', "w")
#writing_file.write(new_file_content)
#writing_file.close()


#previous write

## #w - write and replace  #a - append
#with open(sys.path[0] + '/output/results.csv', 'a', newline='') as f:
#    writer = csv.writer(f)
#
#    writer.writerow([best_result.dataset_name, str_balancing + best_result.algorithm])
#    writer.writerow(["accuracy_score", str(best_result.accuracy)])
#    writer.writerow(["f1_score", str(best_result.f1_score)])
#    writer.writerow(["roc_auc_score", str(best_result.roc_auc_score)])
#    writer.writerow(["time", str(best_result.time)])
#    writer.writerow(["---"])
    
print("fim")