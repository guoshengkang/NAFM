import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import basicFM, DeepFM, NFM, AFM, NAFM
from deepctr.inputs import  SparseFeat, DenseFeat, get_feature_names

if __name__ == "__main__":
    # load input data
    with open('input_data.pickle', 'rb') as f:
        input_data = pickle.load(f)
    dnn_feature_columns=input_data["dnn_feature_columns"]
    linear_feature_columns=input_data["linear_feature_columns"]
    train=input_data["train"]
    test=input_data["test"]
    train_model_input=input_data["train_model_input"]
    test_model_input=input_data["test_model_input"]
    target = ['label']

    # default velues of parameters
    [dnn_hidden_units,attention_factor,l2_reg,dropout,batch_size]=[64, 32, 0.001, 0.1, 256]

    p_dnn_hidden_units=[4,8,16,32,64,128,256]
    p_attention_factor=[4,8,16,32,64,128,256]
    p_l2_reg=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
    p_dropout=[0.1,0.2,0.3,0.4,0.5,0.6,0.7]

    df_loss=pd.DataFrame(np.zeros((4,7)),index=["dnn_hidden_units","attention_factor","l2_reg","dropout"])
    df_auc=pd.DataFrame(np.zeros((4,7)),index=["dnn_hidden_units","attention_factor","l2_reg","dropout"])

    # inpact of dnn_hidden_units
    for index, parameter in enumerate(p_dnn_hidden_units):
        dnn_hidden_units=parameter
        LOSS=[];AUC=[]
        for i in range(5):
            model = NAFM(linear_feature_columns, dnn_feature_columns, attention_factor=attention_factor,
            	dnn_hidden_units=(dnn_hidden_units,dnn_hidden_units),l2_reg_linear=l2_reg, l2_reg_embedding=l2_reg, 
            	l2_reg_dnn=l2_reg,l2_reg_att=l2_reg, dnn_dropout=dropout,afm_dropout=dropout,bi_dropout=dropout,task='binary')
            model.compile("adam", "binary_crossentropy",
                      metrics=['binary_crossentropy'], )
            history = model.fit(train_model_input, train[target].values,
                            batch_size=batch_size, epochs=100, verbose=2, validation_split=0.2, )
            pred_ans = model.predict(test_model_input, batch_size=256)
            loss_value=log_loss(test[target].values, pred_ans)
            auc_value=roc_auc_score(test[target].values, pred_ans)
            LOSS.append(loss_value); AUC.append(auc_value)
        df_loss.ix["dnn_hidden_units",index]=round(np.mean(LOSS),4)
        df_auc.ix["dnn_hidden_units",index]=round(np.mean(AUC),4)

    # inpact of attention_factor
    for index, parameter in enumerate(p_attention_factor):
        attention_factor=parameter
        LOSS=[];AUC=[]
        for i in range(5):
            model = NAFM(linear_feature_columns, dnn_feature_columns, attention_factor=attention_factor,
                dnn_hidden_units=(dnn_hidden_units,dnn_hidden_units),l2_reg_linear=l2_reg, l2_reg_embedding=l2_reg, 
                l2_reg_dnn=l2_reg,l2_reg_att=l2_reg, dnn_dropout=dropout,afm_dropout=dropout,bi_dropout=dropout,task='binary')
            model.compile("adam", "binary_crossentropy",
                      metrics=['binary_crossentropy'], )
            history = model.fit(train_model_input, train[target].values,
                            batch_size=batch_size, epochs=100, verbose=2, validation_split=0.2, )
            pred_ans = model.predict(test_model_input, batch_size=256)
            loss_value=log_loss(test[target].values, pred_ans)
            auc_value=roc_auc_score(test[target].values, pred_ans)
            LOSS.append(loss_value); AUC.append(auc_value)
        df_loss.ix["attention_factor",index]=round(np.mean(LOSS),4)
        df_auc.ix["attention_factor",index]=round(np.mean(AUC),4)


    # inpact of l2_reg
    for index, parameter in enumerate(p_l2_reg):
        l2_reg=parameter
        LOSS=[];AUC=[]
        for i in range(5):
            model = NAFM(linear_feature_columns, dnn_feature_columns, attention_factor=attention_factor,
                dnn_hidden_units=(dnn_hidden_units,dnn_hidden_units),l2_reg_linear=l2_reg, l2_reg_embedding=l2_reg, 
                l2_reg_dnn=l2_reg,l2_reg_att=l2_reg, dnn_dropout=dropout,afm_dropout=dropout,bi_dropout=dropout,task='binary')
            model.compile("adam", "binary_crossentropy",
                      metrics=['binary_crossentropy'], )
            history = model.fit(train_model_input, train[target].values,
                            batch_size=batch_size, epochs=100, verbose=2, validation_split=0.2, )
            pred_ans = model.predict(test_model_input, batch_size=256)
            loss_value=log_loss(test[target].values, pred_ans)
            auc_value=roc_auc_score(test[target].values, pred_ans)
            LOSS.append(loss_value); AUC.append(auc_value)
        df_loss.ix["l2_reg",index]=round(np.mean(LOSS),4)
        df_auc.ix["l2_reg",index]=round(np.mean(AUC),4)

    # inpact of dropout
    for index, parameter in enumerate(p_dropout):
        dropout=parameter
        LOSS=[];AUC=[]
        for i in range(5):
            model = NAFM(linear_feature_columns, dnn_feature_columns, attention_factor=attention_factor,
                dnn_hidden_units=(dnn_hidden_units,dnn_hidden_units),l2_reg_linear=l2_reg, l2_reg_embedding=l2_reg, 
                l2_reg_dnn=l2_reg,l2_reg_att=l2_reg, dnn_dropout=dropout,afm_dropout=dropout,bi_dropout=dropout,task='binary')
            model.compile("adam", "binary_crossentropy",
                      metrics=['binary_crossentropy'], )
            history = model.fit(train_model_input, train[target].values,
                            batch_size=batch_size, epochs=100, verbose=2, validation_split=0.2, )
            pred_ans = model.predict(test_model_input, batch_size=256)
            loss_value=log_loss(test[target].values, pred_ans)
            auc_value=roc_auc_score(test[target].values, pred_ans)
            LOSS.append(loss_value); AUC.append(auc_value)
        df_loss.ix["dropout",index]=round(np.mean(LOSS),4)
        df_auc.ix["dropout",index]=round(np.mean(AUC),4)

    # save resutls to files
    df_loss.to_csv("Logloss_impact_of_parameters.csv",index=True,float_format='%.4f')
    df_auc.to_csv("AUC_impact_of_parameters.csv",index=True,float_format='%.4f')

