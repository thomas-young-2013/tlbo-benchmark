import sys
import pickle
file_name = sys.argv[1]
tmp_file = 'data/xgb_metadata/%s' % file_name
f = open(tmp_file, 'rb')
data = pickle.load(f)
print('the number of runs is', len(data))
for item in data:
    print('validation acc', item[2])
    print('configurations', item[0])
    print('conf vector', item[1])
    print('-'*100)
