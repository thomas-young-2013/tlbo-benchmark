import pandas as pd

def load_poker():
    L = []
    file_path = 'data/xgb_dataset/poker/poker'
    with open(file_path, 'r') as f:
        for line in f.readlines():
            items = line.split('\n')[0].split(' ')
            d ={}
            d['label'] = int(items[0])
            del items[0]
            for item in items:
                key, value = item.split(':')
                d[key] = value
            L.append(d)
        df = pd.DataFrame(L)
        y = df['label'].values
        del df['label']
        X = df.values
        return X, y
if __name__ == '__main__':
    X, y = load_poker()
    print(set(y))