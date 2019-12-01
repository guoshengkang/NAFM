#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-27 14:47:31
# @Author  : Guosheng Kang (guoshengkang@gmail.com)
# @Link    : https://guoshengkang.github.io
# @Version : $Id$

import os
import sys
import csv
import string
import pickle
from collections import Counter
from random import shuffle
import nltk
nltk.download('stopwords')  # run once
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()  # lemmatizer e.g., was-->be
from nltk.stem.snowball import SnowballStemmer
englishStemmer=SnowballStemmer("english") # word stemming e.g., having-->have
import gensim
from gensim.models.doc2vec import Doc2Vec, LabeledSentence #从gensim导入doc2vec
TaggededDocument = gensim.models.doc2vec.TaggedDocument

def sentence_nomalized_split(input_string=None):
	"""
	input:英文文本
	output:英文单词的分词
	"""
	word_list=gensim.utils.simple_preprocess(input_string) # 转成小写，去掉数字，标点符号，长度为1的字符
	split_list=[]
	for word in word_list:
		if word in stop_words:
			continue
		word=wordnet_lemmatizer.lemmatize(word, pos="v")  # lemmatizer e.g., was-->be
		word=englishStemmer.stem(word)  # word stemming e.g., having-->have
		split_list.append(word)
	return split_list

def read_Mashup_API_file(file_path=None):
	# 读取Mashups.csv 或者 APIs.csv文件,将Mashup数据保存到一个Mashups列表中,其中每个元素为一个字典
	with open(file_path, 'r',encoding="utf-8",errors='ignore') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
		All_in_one=[]
		for no,row in enumerate(spamreader):
			one_dict={} # 临时字典
			if no==0:
				csv_head_list=row
				csv_head_list.pop() #去除最后一个元素'\n'
				continue
			else:
				for index,field in enumerate(csv_head_list):
					if field=="MemberAPIs":
						Called_APIs=row[index].split(" @@@ ")
						one_dict[field]=Called_APIs
					elif field=="tags":
						Labeled_tags=row[index].split("###")
						one_dict[field]=Labeled_tags
					elif field=="desc":
						one_dict[field]=sentence_nomalized_split(input_string=row[index])
					else:
						one_dict[field]=row[index]
			All_in_one.append(one_dict)
	return All_in_one,csv_head_list

#Doc2Vec模型训练
def Doc2Vec_train(x_train, size=100, dm=1):  # size=100 意味着每个词向量是100维的
	# 使用 Doc2Vec 建模
	model_dm = Doc2Vec(documents=x_train, dm=dm, size=size, epochs=10, min_count=1, window=3,  sample=1e-3, negative=5, workers=4)
	# dm: 训练算法：默认为1，指PV-DM；dm=0,则使用PV-DBOW
	# size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好.
	# window：表示当前词与预测词在一个句子中的最大距离是多少
	# alpha: 是学习速率
	# min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
	# workers参数控制训练的并行数
	# epochs： 迭代次数，默认为5
	# sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，官网给的解释 1e-5效果比较好。
	# hs: 如果为1则会采用hierarchica·softmax技巧。如果设置为0（默认），则使用negative sampling。 
	model_dm.save("doc2vec.model")
	return model_dm

def get_documents(Mashups=None,APIs=None):
	x_train=[]
	for i, Mashup in enumerate(Mashups):
		word_list=Mashup["desc"]
		document = TaggededDocument(word_list, tags=[i])
		x_train.append(document)
		Mashup["tags_no"]=i
	Mashup_number=len(Mashups)
	for i, API in enumerate(APIs):
		word_list=API["desc"]
		document = TaggededDocument(word_list, tags=[i+Mashup_number])
		x_train.append(document)
		API["tags_no"]=i+Mashup_number
	return x_train, Mashups, APIs

def popuparity(Mashups=None,APIs=None):
	invocations=[]
	for Mashup in Mashups:
		invocations.extend(Mashup["MemberAPIs"])
	invocation_times=Counter(invocations) # 统计每个API调用的次数,没有的默认为0
	category_frequences=dict() # key: category_name, value: 该category下所有API的frequence值
	for API in APIs:
		API["frequence"]=invocation_times[API["APIName"]]
		frequence_list=category_frequences.get(API["primary_category"],[])
		frequence_list.append(API["frequence"])
		category_frequences[API["primary_category"]]=frequence_list
		MemberMashups=[]
		for Mashup in Mashups:
			if API["APIName"] in Mashup["MemberAPIs"]:
				MemberMashups.append(Mashup["MashupName"])
		API["MemberMashups"]=MemberMashups

	for  API in APIs:
		max_fre=max(category_frequences[API["primary_category"]])
		min_fre=min(category_frequences[API["primary_category"]])
		if max_fre==0:
			API["popularity"]=0
		else:
			API["popularity"]=(API["frequence"]-min_fre)/(max_fre-min_fre)
	return 	APIs

def get_popular_APIs_Mashups(Mashups=None,APIs=None,top_k=100):
	# get most 50 popular APIs
	sorted_index=sorted(range(len(APIs)), key=lambda k: APIs[k]["popularity"], reverse=True)
	popular_APIs=[]
	for index in sorted_index[0:top_k]:
		popular_APIs.append(APIs[index])
	MashupNames=[]
	for API in popular_APIs:
		MashupNames.extend(API["MemberMashups"])
	popular_Mashups=[]
	for MashupName in set(MashupNames):
		for Mashup in Mashups:
			if MashupName==Mashup["MashupName"]:
				popular_Mashups.append(Mashup)
				break
	print("Popular Number of APIs:",len(popular_APIs))
	print("Popular Number of Mashups:",len(popular_Mashups))
	return popular_APIs,popular_Mashups

def get_samples(Mashups,APIs,sample_number=5000):
	all_samples=[]
	for API in APIs:
		for Mashup in Mashups:
			if Mashup["MashupName"] in API["MemberMashups"]:
				label=1
			else:
				label=0
			temp_sample=(API,Mashup,label)
			all_samples.append(temp_sample)
	positive_samples=[sample for sample in all_samples if sample[2]==1]
	negative_samples=[sample for sample in all_samples if sample[2]==0]
	shuffle(negative_samples)
	Number_of_positive_samples=len(positive_samples)
	Number_of_negative_samples=sample_number-Number_of_positive_samples
	samples=positive_samples
	samples.extend(negative_samples[:Number_of_negative_samples])
	print("Number_of_positive_samples:",Number_of_positive_samples)
	print("Number_of_negative_samples:",Number_of_negative_samples)
	return samples

if __name__ == "__main__":
	home_path = os.path.dirname(os.path.abspath(__file__))
	Mashup_file_path = os.path.join(home_path, 'dataset\\Mashups.csv')
	API_file_path = os.path.join(home_path, 'dataset\\APIs.csv')
	Mashups,Mashups_head_list=read_Mashup_API_file(Mashup_file_path)
	APIs,APIs_head_list=read_Mashup_API_file(API_file_path)
	print(Mashups_head_list,len(Mashups_head_list),len(Mashups[0]))
	print(APIs_head_list,len(APIs_head_list),len(APIs[0]))
	x_train, Mashups, APIs=get_documents(Mashups,APIs)
	model_dm=Doc2Vec_train(x_train, size=50, dm=1)
	# model_dm=Doc2Vec.load("doc2vec.model")

	APIs=popuparity(Mashups,APIs)
	popular_APIs,popular_Mashups=get_popular_APIs_Mashups(Mashups,APIs,top_k=100)
	invocations=[]
	for API in popular_APIs:
		invocations.extend(API["MemberMashups"])
	print("Number of invocatio:",len(invocations))
	categories=[]
	for API in popular_APIs:
		categories.append(API["primary_category"])
	for Mashup in popular_Mashups:
		categories.append(Mashup["primary_category"])
	print("Number of categories for Mashups and APIs:",len(set(categories)))
	samples=get_samples(popular_Mashups,popular_APIs,sample_number=5000)
	# 保存
	with open('samples.pickle', 'wb') as f:
	    pickle.dump(samples, f)
	# # 读取
	# with open('samples.pickle', 'rb') as f:
	#     samples = pickle.load(f)



