import pandas as pd

def load_smallNORB():
    L = []
    file_path = '../../../data/xgb_dataset/smallNORB/smallNORB'
    with open(file_path, 'r') as f:
        for line in f.readlines():
            items = line.strip().split('\n')[0].split(' ')
            d ={}
            d['label'] = int(items[0])
            del items[0]
            for item in items:
                key, value = item.split(':')
                d[key] = int(value)
            L.append(d)
        df = pd.DataFrame(L)
        y = df['label'].values
        del df['label']
        df.fillna(0,inplace=True)
        X = df.values
        return X, y
if __name__ == '__main__':
    X, y = load_smallNORB()
    print(X)
    print(y)