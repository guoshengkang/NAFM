import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.inputs import  SparseFeat, DenseFeat, get_feature_names

if __name__ == "__main__":
    data = pd.read_csv('./input_data.csv')

    sparse_features = ['C' + str(i) for i in range(1, 3)]
    dense_features = ['I' + str(i) for i in range(1, 103)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4)
                           for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                          for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train, test = train_test_split(data, test_size=0.2)
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}

    # save input data
    # put them together
    input_data={"dnn_feature_columns":dnn_feature_columns,
    "linear_feature_columns":linear_feature_columns,
    "train":train, "test":test,
    "train_model_input":train_model_input,
    "test_model_input":test_model_input}
    with open('input_data.pickle', 'wb') as f:
        pickle.dump(input_data, f)

