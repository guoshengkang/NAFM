#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-14 20:52:25
# @Author  : Guosheng Kang (guoshengkang@gmail.com)
# @Link    : https://guoshengkang.github.io
# @Version : $Id$

import os
import sys
import csv
import string

home_path = os.path.dirname(os.path.abspath(__file__))
Mashup_file_path = os.path.join(home_path, 'dataset\\Mashups.csv')
API_file_path = os.path.join(home_path, 'dataset\\APIs.csv')

def English_sentence_cut(input_string):
	"""
	input:英文文本
	output:英文单词的分词
	"""
	input_string=input_string.lower() #将字母统一换成小写
	for char in string.punctuation: # 将标点符号替换为空格
		input_string=input_string.replace(char, " ")
	split_list=input_string.split() # 默认split按一个或多个空格分隔
	return split_list

# 读取Mashups.csv文件,将Mashup数据保存到一个Mashups列表中,其中每个元素为一个字典
with open(Mashup_file_path, 'r',encoding="utf-8",errors='ignore') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
	Mashups=[]
	invocations=[]
	Mashup_tags=[]
	for no,row in enumerate(spamreader):
		Mashup_dict={} # 临时字典
		if no==0:
			csv_head_list=row
			csv_head_list.pop() #去除最后一个元素'\n'
			continue
		else:
			for index,field in enumerate(csv_head_list):
				if field=="MemberAPIs":
					Called_APIs=row[index].split(" @@@ ")
					Mashup_dict[field]=Called_APIs
					invocations.extend(Called_APIs)
				elif field=="tags":
					Labeled_tags=row[index].split("###")
					Mashup_dict[field]=Labeled_tags
					Mashup_tags.extend(Labeled_tags)
				else:
					Mashup_dict[field]=row[index]
		Mashups.append(Mashup_dict)

# 读取APIs.csv文件,将Mashup数据保存到一个APIs列表中,其中每个元素为一个字典
with open(API_file_path, 'r',encoding="utf-8",errors='ignore') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
	APIs=[]
	API_tags=[]
	for no,row in enumerate(spamreader):
		API_dict={} # 临时字典
		if no==0:
			csv_head_list=row
			# csv_head_list.pop() #去除最后一个元素'\n'
			continue
		else:
			for index,field in enumerate(csv_head_list):
				if field=="tags":
					Labeled_tags=row[index].split("###")
					API_dict[field]=Labeled_tags
					API_tags.extend(Labeled_tags)
				else:
					API_dict[field]=row[index]
		APIs.append(API_dict)


# 统计Mashup与API的相关数据
Number_of_Mashups=len(Mashups) # Mashup 的数量
Number_of_APIs=len(APIs) # API 的数量
Number_of_invocations=len(invocations) #Mashup与API之间的调用数量
Average_number_of_invocations_per_Mashup=Number_of_invocations/Number_of_Mashups # 每个Mashup的平均调用数量
# 注意：Mashup调用的部分API不在此API数据集中
APINames=[]
for API in APIs:
	APIName=API["APIName"]
	APINames.append(APIName)
interactions=[]
for invocation in invocations:
	if invocation in APINames:
		interactions.append(invocation)
Number_of_called_APIs=len(set(interactions)) # 被调用的Web API数量
Called_proportion_of_APIs=Number_of_called_APIs/Number_of_APIs # 被调用API的比例
Number_of_interactions=len(interactions) # Mashup与包含的API之间的调用次数
Number_of_labeled_Mashup_tags=len(Mashup_tags) # Mashup中包含的Tag数量
Number_of_Mashup_tags=len(set(Mashup_tags)) # 被使用的Mashup Tag数量
Number_of_labeled_API_tags=len(API_tags) # API中包含的Tag数量
Number_of_API_tags=len(set(API_tags)) # 被使用的API Tag数量

Mashup_length=0
for Mashup in Mashups:
	Mashup_length=Mashup_length+len(English_sentence_cut(input_string=Mashup["desc"]))
Average_length_of_Mashup_description=Mashup_length/len(Mashups) # Mashup的平均描述长度
API_length=0
for API in APIs:
	API_length=API_length+len(English_sentence_cut(input_string=API["desc"]))
Average_length_of_API_description=API_length/len(APIs) # API的平均描述长度

#如果Mashup调用的API不在数据集中,则矩阵中将其去掉
Number_of_Mashups_with_included_APIs=0
for Mashup in Mashups:
	MemberAPIs=Mashup["MemberAPIs"]
	Flag=0
	for API in MemberAPIs:
		if API in APINames:
			Flag=1
			Number_of_Mashups_with_included_APIs=Number_of_Mashups_with_included_APIs+1
			break	

sparsity=1-Number_of_interactions/(Number_of_Mashups_with_included_APIs*Number_of_called_APIs) # Mashup-API矩阵的稀疏度

# 统计数据输出
print("Number of Mashups:",Number_of_Mashups)
print("Number of APIs:",Number_of_APIs)
print("Number of invocations:",Number_of_invocations)
print("Average number of invocations per Mashup: %.2f" % Average_number_of_invocations_per_Mashup)
print("Number of called APIs:",Number_of_called_APIs)
print("Called proportion of APIs: %.2f%%" % (Called_proportion_of_APIs*100))
print("NUmber of interactions: ",Number_of_interactions)
print("Number of labeled Mashup tags:",Number_of_labeled_Mashup_tags)
print("Number of Mashup tags:",Number_of_Mashup_tags)
print("Number of labeled API tags:",Number_of_labeled_API_tags)
print("Number of API tags:",Number_of_API_tags)
print("Average length of Mashup description: %.2f" % Average_length_of_Mashup_description)
print("Average length of API description: %.2f" % Average_length_of_API_description)
print("Number of Mashups with included APIs:",Number_of_Mashups_with_included_APIs)
print("Sparsity of Mashup-API matrix: %.2f%%" % (sparsity*100))