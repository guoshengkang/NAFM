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

# 读取Mashups.csv文件
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

# 统计Mashup相关数据
Number_of_Mashups=len(Mashups)
print("Number_of_Mashups:",Number_of_Mashups)

Number_of_invocations=len(invocations)
print("Number_of_invocations:",Number_of_invocations)
Number_of_called_APIs=len(set(invocations)) # 被调用的Web API数量
print("Number_of_called_APIs:",Number_of_called_APIs)

Number_of_labeled_Mashup_tags=len(Mashup_tags)
print("Number_of_labeled_Mashup_tags:",Number_of_labeled_Mashup_tags)
Number_of_Mashup_tags=len(set(Mashup_tags)) # 被使用的Mashup Tag数量
print("Number_of_Mashup_tags:",Number_of_Mashup_tags)

Mashup_length=0
for Mashup in Mashups:
	deccription=Mashup["desc"].lower()
	for char in string.punctuation:
		deccription=deccription.replace(char, " ")
	word_list=deccription.split() # 默认split按一个或多个空格分隔
	Mashup_length=Mashup_length+len(word_list)
Average_length_of_Mashup_description=Mashup_length/len(Mashups)
print("Average_length_of_Mashup_description:%.4f" % Average_length_of_Mashup_description)

sparsity=1-Number_of_invocations/(Number_of_Mashups*Number_of_called_APIs)
print("Sparsity_of_Mashup-API_matrix:%.4f" % sparsity)

# 读取APIs.csv文件
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

Number_of_APIs=len(APIs)
print("Number_of_APIs:",Number_of_APIs)

# 统计API相关数据
API_length=0
for API in APIs:
	deccription=API["descr"].lower()
	for char in string.punctuation:
		deccription=deccription.replace(char, " ")
	word_list=deccription.split() # 默认split按一个或多个空格分隔
	API_length=API_length+len(word_list)
Average_length_of_API_description=API_length/len(APIs)
print("Average_length_of_API_description:%.4f" % Average_length_of_API_description)
Called_proportion_of_APIs=Number_of_called_APIs/Number_of_APIs
print("Called_proportion_of_APIs:%.4f" % Called_proportion_of_APIs)
