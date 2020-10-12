import sys
import pickle
dataset_name = sys.argv[1]
tmp_file = 'data/tmp_%s_result.pkl' % dataset_name
f = open(tmp_file, 'rb')
data = pickle.load(f)
print('the number of runs is', len(data))
for item in data:
    print('validation error', item[2])
    print('test error', item[3])
    print('configurations', item[0], item[1])
    print('-'*100)
