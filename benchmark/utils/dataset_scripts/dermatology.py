import numpy as np

def load_dermatology():
    L = []
    file_path = 'data/xgb_dataset/dermatology/dermatology.txt'
    with open(file_path, 'r') as f:
        for line in f.readlines():
            items = line.split('\n')[0].split(',')
            l = []
            for i in items:
                l.append(int(i))
            L.append(l)
    data = np.array(L)
    return data[:,1:], data[:,0]

if __name__ == '__main__':
    print(load_dermatology())
