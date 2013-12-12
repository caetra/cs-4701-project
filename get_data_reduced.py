# Script to read in the spambase.data files and get a format we can use

import pickle
import csv

filename = 'spambase.test'

f = open(filename, 'r')

data = []
freader = csv.reader(f, delimiter=',')
for row in freader:
    data.append(row)
    
f.close()

#print data

# There are 58 columns in the data...
# 57 are attributes
# 48 are frequency of a word
# 6 are frequency of a punctuation mark
# 3 are data on sequences of characters
# Last is label
num_attributes = 57

attributes = []
labels = []
for example in data:
    #attributes.append(example[:num_attributes])
    attributes.append(example[:54])
    labels.append(example[num_attributes])
    
# convert the strings to numbers
for row in range(len(attributes)):
    for col in range(54):
        attributes[row][col] = float(attributes[row][col])
        
for x in range(len(labels)):
    labels[x] = float(labels[x])
    
print attributes[0]
attr_out = open('test_data_reduced.pickle', 'w')
label_out = open('test_label.pickle', 'w')

pickle.dump(attributes, attr_out)
pickle.dump(labels, label_out)

attr_out.close()
label_out.close()