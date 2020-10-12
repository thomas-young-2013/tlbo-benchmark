## Description
Tuning task descriptions for creating this benchmark.

## Hyperparameters for ResNet

| hp | lower | upper | default | scale | q |
| :-- | :-- | :-- | :-- | :-- | :-- |
| pooling_type | - | - | 'avg'| - | enumerate{'avg', 'max'} |
| lr | 0.001 | 0.5 | 0.1 | log | 0.003 |
| momentum | 0 | 0.95 | 0 | uniform | 0.05 |
| decay | 0 | 1e-3 | 1e-4 | uniform | 3e-6 |
| nesterov | - | - | True | - | enumerate{True, False} |
| batch_size | 16 | 256 | 64 | uniform | 16 |
| dropout | 0 | 0.6 | 0 | uniform | 0.05 |
| regularizer | 0 | 1e-3 | 0 | log | 3e-5 |
| lr_reductions | 0.1 | 0.5 | 0.1 | uniform | 0.1 |


## Dataset for ResNet

| dataset | cls_num | instance_num | address | spec |
| :-- | :-- | :-- | :-- | :-- |
| SVHN | 10 | 73257/26032 | [URL](http://ufldl.stanford.edu/housenumbers/) | 50/30 |
| CIFAR-10 | 10 | 50000/10000 | [URL](https://keras.io/datasets/) | 165/30 |
| CIFAR-100 | 100 | 50000/10000 | [URL](https://keras.io/datasets/) | 200/30 |
| Tiny Imagenet | 200 | 98179/9832 | [URL](https://tiny-imagenet.herokuapp.com/) | 200/30 |
| Caltech 101 | 102 | 9145 | [URL](https://keras.io/datasets/) | 100/30 |
| Caltech 256 | 257 | 30607 | [URL](https://keras.io/datasets/) | 200/30 |
| Chars74K (GoodImg) | 62 | 7705 | [URL](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) | 100/30 |
| Plant-Seedlings-Classification(Kaggle) | 12 | 4750 | [URL](https://www.kaggle.com/c/plant-seedlings-classification) | 50/30 |
| Dog Breed Identification(Kaggle) | 120 | 10222 | [URL](https://www.kaggle.com/c/dog-breed-identification/data) | 150/30 |
| Dogs vs Cats | 2 | 25000 | [URL](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) | 50/30 |


---

## Hyperparameters for XGB
#### Booster Parameters

0. **n_estimators**
1. **eta**: learning rate
2. **min_child_weight**: minimum sum of weights of all observations required in a child
3. **max_depth**: the maximum depth of a tree
4. max_leaf_nodes: ignored, as a substitute of max_depth 
5. **gamma**: the minimum loss reduction required to make a split.
6. max_delta_step: each tree’s weight estimation to be, only needed in logistic regression when class is extremely imbalanced.
7. **subsample**: 
8. **colsample_bytree**: 0.3 - 0.8 if many columns; else 0.8 - 1
9. colsample_bylevel
10. **lambda**: L2 regularization term on weights
11. alpha: L1 regularization term on weight, could be a substitute of L2 in case of high dimensinality so that algo runs faster.
12. **scale_pos_weight**: in case of imbalance

| hp | lower | upper | default | scale | q |
| :-- | :-- | :-- | :-- | :-- | :-- |
| n_estimators | 100 | 600 | 200 | uniform | 50 |
| eta | 0.025 | 0.3 | 0.1 | uniform | 0.025 |
| min_child_weight | 1 | 10 | 1 | uniform | - |
| max_depth | 1 | 14 | 6 | uniform | - |
| gamma | 0 | 1 | 0 | uniform | 0.1 |
| subsample | 0.5 | 1 | 1 | uniform | 0.05 |
| colsample_bytree | 0.5 | 1 | 1 | uniform | 0.05 |
| alpha | 0 | 10 | 0 | uniform | 1 |
| lambda | 1 | 2 | 1 | uniform | 0.1 |
| scale_pos_weight | 0.01,0.1,1,10,100 | - | 1. | - | - |

## Dataset for XGB
1. Kaggle competition: [Fall Detection Data from China](https://www.kaggle.com/pitasr/falldata)
2. Kaggle competition (two tasks): [Biomechanical features of orthopedic patients](Biomechanical features of orthopedic patients
)
3. Kaggle competition: [Standard Classification (Banana Dataset)](https://www.kaggle.com/saranchandar/standard-classification-banana-dataset)
4. Kaggle competition: [TalkingData AdTracking Fraud Detection Challenge](https://www.kaggle.com/alexandrnikitin/xgboost-hyperparameter-optimization/data)
5. susy at R2
6. higgs at R2
7. hepmass at R2
8. Epsilon at R1
9. Usps at R1
10. W8a at R1
10. A8a at R1
11. Letter at R1
12. dermatology at R2
13. pima-indians-diabetes at R2 \
Thank you for your interest in the Pima Indians Diabetes dataset. 
The dataset is no longer available due to permission restrictions.
14. phishing at R1
15. poker at R1
16. sector at R1
17. protein at R1
18. sensorless at R1
19. shuttle at R1
20. smallNORB at R1

Resource Websites:
1. [R1](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)
2. [R2](https://archive.ics.uci.edu/ml/datasets.html)

# Attentions Here
1. the label in UCI data file may need be processed: change [1-10] to [0-9]
2. the missing value in some columns.
```
data = np.loadtxt('./dermatology.data', delimiter=',',
        converters={33: lambda x:int(x == '?'), 34: lambda x:int(x) - 1})
sz = data.shape
```

## Building Status
| dataset | source | description | status |
| :-- | :-- | :-- | :-- |
| iris | sklearn | 3 labels | ok |
| digits | sklearn | 10 labels | ok |
| wine | sklearn | 3 labels | ok |
| breast_cancer | sklearn | 2 labels | ok |
| olivetti_faces | sklearn | 40 labels | ok |
| 20newsgroups | sklearn | 20 labels | ok |
| 20newsgroups_vectorized | sklearn | 20 labels | Running[140] |
| lfw_people | sklearn | 5749 labels | ok |
| covtype | sklearn | 7 labels | ok |
| rcv1 | sklearn | 103 labels, train| Running[131] |
| kddcup99-smtp | sklearn | 23 labels | ok |
| kddcup99-sa | sklearn | 23 labels | ok |
| fall_detection | kaggle | 6 labels | ok |
| banana | sklearn | 2 labels | ok |
| talkingdata | sklearn | 2 labels | ok |
| biomechanical2C | sklearn | 2 labels | ok |
| biomechanical3C | sklearn | 3 labels | ok |
| susy | sklearn | 2 labels | ok |
| higgs | sklearn | 2 labels | ok |
| hepmass | sklearn | 2 labels | ok |
| letter | sklearn | 26 labels | ok |
| usps | sklearn | 10 labels | ok |
| epsilon | sklearn | 2 labels | ok |
| W8a | sklearn | 2 labels | ok |
| A8a | sklearn | 2 labels | ok |
| dermatology | uci | 6 labels | ok |
| phishing | R1 | 2 labels | ok |
| poker | R1 | 10 labels | ok |
| sector | R1 | 105 labels | so damn slow! give it up QAQ |
| protein | R1 | 3 labels | ok |
| sensorless | R1 | 11 labels | ok |
| shuttle | R1 | 7 labels | ok |
| vowel | r1 | 11 labels | ok |
| splice | r1 | 2 labels | ok |
| cod-rna | r1 | 2 labels | ok |
| australian | r1 | 2 labels | ok |