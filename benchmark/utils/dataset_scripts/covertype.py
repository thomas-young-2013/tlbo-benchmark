import gc
import numpy as np
import pandas as pd
import xgboost as xgb

from pandas.core.categorical import Categorical
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split

def sparse_dummies(df, column):
    '''Returns sparse OHE matrix for the column of the dataframe'''
    categories = Categorical(df[column])
    column_names = np.array(["{}_{}".format(column, str(i)) for i in range(len(categories.categories))])
    N = len(categories)
    row_numbers = np.arange(N, dtype=np.int)
    ones = np.ones((N,))
    return csr_matrix((ones, (row_numbers, categories.codes))), column_names

column_names = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points','Wilderness_Area_0','Wilderness_Area_1','Wilderness_Area_2','Wilderness_Area_3','Soil_Type_0','Soil_Type_1','Soil_Type_2','Soil_Type_3','Soil_Type_4','Soil_Type_5','Soil_Type_6','Soil_Type_7','Soil_Type_8','Soil_Type_9','Soil_Type_10','Soil_Type_11','Soil_Type_12','Soil_Type_13','Soil_Type_14','Soil_Type_15','Soil_Type_16','Soil_Type_17','Soil_Type_18','Soil_Type_19','Soil_Type_20','Soil_Type_21','Soil_Type_22','Soil_Type_23','Soil_Type_24','Soil_Type_25','Soil_Type_26','Soil_Type_27','Soil_Type_28','Soil_Type_29','Soil_Type_30','Soil_Type_31','Soil_Type_32','Soil_Type_33','Soil_Type_34','Soil_Type_35','Soil_Type_36','Soil_Type_37','Soil_Type_38','Soil_Type_39', 'label']
categorical_features = []
numerical_features = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points','Wilderness_Area_0','Wilderness_Area_1','Wilderness_Area_2','Wilderness_Area_3','Soil_Type_0','Soil_Type_1','Soil_Type_2','Soil_Type_3','Soil_Type_4','Soil_Type_5','Soil_Type_6','Soil_Type_7','Soil_Type_8','Soil_Type_9','Soil_Type_10','Soil_Type_11','Soil_Type_12','Soil_Type_13','Soil_Type_14','Soil_Type_15','Soil_Type_16','Soil_Type_17','Soil_Type_18','Soil_Type_19','Soil_Type_20','Soil_Type_21','Soil_Type_22','Soil_Type_23','Soil_Type_24','Soil_Type_25','Soil_Type_26','Soil_Type_27','Soil_Type_28','Soil_Type_29','Soil_Type_30','Soil_Type_31','Soil_Type_32','Soil_Type_33','Soil_Type_34','Soil_Type_35','Soil_Type_36','Soil_Type_37','Soil_Type_38','Soil_Type_39']

data = np.loadtxt('../../../data/covertype/train.txt', dtype=int, delimiter=',', converters={54: lambda x: int(x) - 1})
# df_train = pd.DataFrame(data, columns=column_names)
# print(df_train)

X = data[:,0:54]
y = data[:,54]
print(X)
print(y)

# del df_train
# gc.collect()

x1, x2, y1, y2 =train_test_split(X, y,test_size=0.2, random_state=0)

# Create binary training and validation files for XGBoost
dm1 = xgb.DMatrix(x1, y1, feature_names=numerical_features)
dm1.save_binary('../../../data/covertype/train_sample.bin')
del dm1, x1, y1
gc.collect()

dm2 = xgb.DMatrix(x2, y2, feature_names=numerical_features)
dm2.save_binary('../../../data/covertype/validate_sample.bin')
del dm2, x2, y2
gc.collect()

f = open('../../../data/covertype/feature_names.txt', 'w')
for i in numerical_features:
    f.write(str(i))
    f.write('\n')
f.close()

