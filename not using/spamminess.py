# We can roughly measure how "spammy" a feature is by finding how different it is between an average spam and an average ham.

import pickle

f1 = open('test_data.pickle','r')
f2 = open('test_label.pickle','r')
data = pickle.load(f1)
labels = pickle.load(f2)

outfile = open('spamminess.txt','w')

spam_sums = [0]*57
ham_sums = [0]*57
for i in range(1601):
    if labels[i] == 1:
        for j in range(57):
            spam_sums[j] += data[i][j]
    else:
        for j in range(57):
            ham_sums[j] += data[i][j]
            
for i in range(57):
    outfile.write('attribute ' +str(i+1)+ ': ' +str((spam_sums[i]/1601.0)-(ham_sums[i]/1601.0))+'\n')

f1.close()
f2.close()
outfile.close()