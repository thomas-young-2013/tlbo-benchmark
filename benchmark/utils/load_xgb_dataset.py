import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


def trans_label(input):
    le = preprocessing.LabelEncoder()
    le.fit(input)
    return le.transform(input)


def one_hot(input):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(input)
    return enc.transform(input)


def load_data(dataset_name):
    if dataset_name == 'iris':
        from sklearn.datasets import load_iris
        iris = load_iris()
        X, y = iris.data, iris.target
        num_cls = 3
    elif dataset_name == 'digits':
        from sklearn.datasets import load_digits
        digits = load_digits()
        X, y = digits.data, digits.target
        num_cls = 10
    elif dataset_name == 'wine':
        from sklearn.datasets import load_wine
        wine = load_wine()
        X, y = wine.data, wine.target
        num_cls = 3
    elif dataset_name == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X, y = data.data, data.target
        num_cls = 2
    elif dataset_name == 'olivetti_faces':
        from sklearn.datasets.olivetti_faces import fetch_olivetti_faces
        data = fetch_olivetti_faces()
        X, y = data.data, data.target
        num_cls = 40
    elif dataset_name == '20newsgroups':
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.datasets.twenty_newsgroups import fetch_20newsgroups
        data = fetch_20newsgroups()
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(data.data)
        y = data.target
        num_cls = 20
    elif dataset_name == '20newsgroups_vectorized':
        from sklearn.datasets.twenty_newsgroups import fetch_20newsgroups_vectorized
        data = fetch_20newsgroups_vectorized()
        X, y = data.data, data.target
        num_cls = 20
    elif dataset_name == 'lfw_people':
        from sklearn.datasets.lfw import fetch_lfw_people
        # all dataset
        # data = fetch_lfw_people()
        # X, y = data.data, data.target
        # num_cls = 5749
        # subset
        data = fetch_lfw_people(min_faces_per_person=50)
        X, y = data.data, data.target
        num_cls = 12
        assert num_cls == len(set(y))
    elif dataset_name == 'covtype':
        from benchmark.utils.dataset_scripts.covtype import load_covtype
        X, y = load_covtype()
        num_cls = 7
    elif dataset_name == 'rcv1':
        from sklearn.datasets.rcv1 import fetch_rcv1
        data = fetch_rcv1(subset='train')
        X, y = data.data, trans_label(np.argmax(data.target.A, axis=1))
        num_cls = len(set(y))
        assert num_cls == 37
    elif dataset_name == 'kddcup99-sa':
        from sklearn.datasets.kddcup99 import fetch_kddcup99
        data = fetch_kddcup99(subset='SA')

        y = trans_label(data.target)
        # one-hot encode the 1-th and 3-th column.
        # col1 = trans_label(data.data[:, 1]).reshape((len(data.data), 1))
        # col1 = one_hot(col1)
        # col3 = trans_label(data.data[:, 3]).reshape((len(data.data), 1))
        # col3 = one_hot(col3)

        # 2-th column has too many categories, just encode the label.
        X = data.data.copy()
        X[:, 2] = trans_label(data.data[:, 2])
        X[:, 1] = trans_label(data.data[:, 1])
        X[:, 3] = trans_label(data.data[:, 3])
        # # print(col1.shape, col3.shape, X.shape)
        # X = np.delete(X, [1, 3], axis=1)
        # X = X.astype(float)
        # print(col1.shape, col3.shape, X.shape)
        # print(X.dtype, col1.dtype, col3.dtype)
        # X = np.c_[X, col1]
        # X = np.c_[X, col3]
        num_cls = len(set(y))
    elif dataset_name == 'kddcup99-smtp':
        from sklearn import preprocessing
        from sklearn.datasets.kddcup99 import fetch_kddcup99
        data = fetch_kddcup99(subset='smtp')
        le = preprocessing.LabelEncoder()
        le.fit(data.target)
        y = le.transform(data.target)
        X = data.data
        num_cls = 3
    elif dataset_name == 'fall_detection':
        from benchmark.utils.dataset_scripts.fall_detection import load_fall_detection
        X, y = load_fall_detection()
        num_cls = 6
    elif dataset_name == 'banana':
        from benchmark.utils.dataset_scripts.banana import load_banana
        X, y = load_banana()
        num_cls = 2
    elif dataset_name == 'talkingdata':
        from benchmark.utils.dataset_scripts.talkingdata import load_talkinigdata
        X, y = load_talkinigdata()
        num_cls = 2
    elif dataset_name == 'biomechanical2C':
        from benchmark.utils.dataset_scripts.biomechanical import load_biomechanical2C
        X, y = load_biomechanical2C()
        num_cls = 2
    elif dataset_name == 'biomechanical3C':
        from benchmark.utils.dataset_scripts.biomechanical import load_biomechanical3C
        X, y = load_biomechanical3C()
        num_cls = 3
    elif dataset_name == 'susy':
        from benchmark.utils.dataset_scripts.susy import load_susy
        X, y = load_susy()
        num_cls = 2
    elif dataset_name == 'higgs':
        from benchmark.utils.dataset_scripts.higgs import load_higgs
        X, y = load_higgs()
        num_cls = 2
    elif dataset_name == 'hepmass':
        from benchmark.utils.dataset_scripts.hepmass import load_hepmass
        X, y = load_hepmass()
        num_cls = 2
    elif dataset_name == 'letter':
        from benchmark.utils.dataset_scripts.letter import load_letter
        X, y = load_letter()
        num_cls = 26
    elif dataset_name == 'usps':
        from benchmark.utils.dataset_scripts.usps import load_usps
        X, y = load_usps()
        num_cls = 10
    elif dataset_name == 'epsilon':
        from benchmark.utils.dataset_scripts.usps import load_epsilon
        X, y = load_epsilon()
        num_cls = 2
    elif dataset_name == 'dermatology':
        from benchmark.utils.dataset_scripts.dermatology import load_dermatology
        X, y = load_dermatology()
        num_cls = 4
    elif dataset_name == 'poker':
        from benchmark.utils.dataset_scripts.poker import load_poker
        X, y = load_poker()
        num_cls = 10
    elif dataset_name == 'sensorless':
        from benchmark.utils.dataset_scripts.sensorless import load_sensorless
        X, y = load_sensorless()
        num_cls = 11
    elif dataset_name == 'phishing':
        from benchmark.utils.dataset_scripts.phishinig import load_phishing
        X, y = load_phishing()
        num_cls = 2
    elif dataset_name == 'w8a':
        from benchmark.utils.dataset_scripts.w8a import load_w8a
        X, y = load_w8a()
        num_cls = 2
    elif dataset_name == 'a8a':
        from benchmark.utils.dataset_scripts.a8a import load_a8a
        X, y = load_a8a()
        num_cls = 2
    elif dataset_name == 'sector':
        from benchmark.utils.dataset_scripts.sector import load_sector
        X, y = load_sector()
        num_cls = 105
    elif dataset_name == 'protein':
        from benchmark.utils.dataset_scripts.protein import load_protein
        X, y = load_protein()
        num_cls = 3
    elif dataset_name == 'shuttle':
        from benchmark.utils.dataset_scripts.shuttle import load_shuttle
        X, y = load_shuttle()
        num_cls = 7
    elif dataset_name == 'vowel':
        from benchmark.utils.dataset_scripts.vowel import load_vowel
        X, y = load_vowel()
        num_cls = 11
    elif dataset_name == 'splice':
        from benchmark.utils.dataset_scripts.splice import load_splice
        X, y = load_splice()
        num_cls = 2
    elif dataset_name == 'codrna':
        from benchmark.utils.dataset_scripts.codrna import load_codrna
        X, y = load_codrna()
        num_cls = 2
    elif dataset_name == 'australian':
        from benchmark.utils.dataset_scripts.australian import load_australian
        X, y = load_australian()
        num_cls = 2
    else:
        raise ValueError('Invalid dataset name!')

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return (x_train, y_train), (x_test, y_test), num_cls
