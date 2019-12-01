#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-28 14:42:35
# @Author  : Guosheng Kang (guoshengkang@gmail.com)
# @Link    : https://guoshengkang.github.io
# @Version : $Id$

import os
import numpy as np
import pandas as pd
import pickle
import gensim
from gensim.models.doc2vec import Doc2Vec, LabeledSentence #从gensim导入doc2vec
TaggededDocument = gensim.models.doc2vec.TaggedDocument

def cos_similarity(arr1=None,arr2=None):
    cos_sim = np.dot(arr1,arr2) / (np.linalg.norm(arr1)*np.linalg.norm(arr2))
    # cos_sim = 0.5 + 0.5*cos_sim # 归一化[0,1]
    return cos_sim

model_dm=Doc2Vec.load("doc2vec.model")

# 读取
with open('samples.pickle', 'rb') as f:
    samples = pickle.load(f)
# headline
columns=[]
label_column=["label"]
API_doc2vec_columns=["I"+str(k) for k in range(1,51)]
Mashup_doc2vec_columns=["I"+str(k) for k in range(51,101)]
API_Mashup_similarity_column=["I101"]
API_popularity_column=["I102"]
API_category_column=["C1"]
Mashup_category_column=["C2"]
all_columns=[label_column,API_doc2vec_columns,Mashup_doc2vec_columns,API_Mashup_similarity_column,API_popularity_column,API_category_column,Mashup_category_column]
for temp_columns in all_columns:
	columns.extend(temp_columns)
print("columns:",columns)

row_number=len(samples)
col_number=len(columns)
print("row_number:",row_number)
print("col_number:",col_number)

df=pd.DataFrame(np.zeros((row_number,col_number)),columns=columns)
for index, sample in enumerate(samples): # (API,Mashup,label)
	df.ix[index,label_column]=str(sample[2])
	API_doc2vec_values=model_dm.infer_vector(sample[0]["desc"]) # API doc2vec 
	df.ix[index,API_doc2vec_columns]=API_doc2vec_values
	Mashup_doc2vec_values=model_dm.infer_vector(sample[1]["desc"]) # Mashup doc2vec 
	df.ix[index,Mashup_doc2vec_columns]=Mashup_doc2vec_values
	df.ix[index,API_Mashup_similarity_column]=model_dm.docvecs.similarity(sample[0]["tags_no"],sample[1]["tags_no"])    #cos_similarity(API_doc2vec_values,Mashup_doc2vec_values)
	df.ix[index,API_popularity_column]=sample[0]["popularity"]
	df.ix[index,API_category_column]=sample[0]["primary_category"]
	df.ix[index,Mashup_category_column]=sample[1]["primary_category"]

# save df to csv
df.to_csv("input_data.csv",index=False,float_format='%.9f')






