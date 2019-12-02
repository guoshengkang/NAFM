#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-12-02 09:29:18
# @Author  : Guosheng Kang (guoshengkang@gmail.com)
# @Link    : https://guoshengkang.github.io
# @Version : $Id$

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

p_dnn_hidden_units=[4,8,16,32,64,128,256]
p_attention_factor=[4,8,16,32,64,128,256]
p_l2_reg=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
p_dropout=[0.1,0.2,0.3,0.4,0.5,0.6,0.7]

df_loss = pd.read_csv('Logloss_impact_of_parameters.csv',index_col=0)
df_auc = pd.read_csv('AUC_impact_of_parameters.csv',index_col=0)

fig=plt.figure(figsize=(13, 6))

# Hidden Units
ax1=fig.add_subplot(2,4,1)
ax1.plot(range(7),df_loss.ix["dnn_hidden_units",],"bv-")
plt.xlabel("Hidden Units\n(a)")
plt.ylabel("Logloss")
plt.xlim((0,6))
plt.xticks(range(7),["4","8","16","32","64","128","256"])
plt.grid(linestyle='-.')

ax2=fig.add_subplot(2,4,5)
ax2.plot(range(7),df_auc.ix["dnn_hidden_units",],"r^-")
plt.xlabel("Hidden Units\n(e)")
plt.ylabel("AUC")
plt.xlim((0,6))
plt.xticks(range(7),["4","8","16","32","64","128","256"])
plt.grid(linestyle='-.')

# Attention Factors
ax1=fig.add_subplot(2,4,2)
ax1.plot(range(7),df_loss.ix["attention_factor",],"bv-")
plt.xlabel("Attention Factors\n(b)")
plt.ylabel("Logloss")
plt.xlim((0,6))
plt.xticks(range(7),["4","8","16","32","64","128","256"])
plt.grid(linestyle='-.')

ax2=fig.add_subplot(2,4,6)
ax2.plot(range(7),df_auc.ix["attention_factor",],"r^-")
plt.xlabel("Attention Factors\n(f)")
plt.ylabel("AUC")
plt.xlim((0,6))
plt.xticks(range(7),["4","8","16","32","64","128","256"])
plt.grid(linestyle='-.')

# L2 Regularization
ax1=fig.add_subplot(2,4,3)
ax1.plot(range(7),df_loss.ix["l2_reg",],"bv-")
plt.xlabel("L2 Regularization\n(c)")
plt.ylabel("Logloss")
plt.xlim((0,6))
plt.xticks(range(7),["1e-7","1e-6","1e-5","1e-4","1e-3","1e-2","1e-1"])
plt.grid(linestyle='-.')

ax2=fig.add_subplot(2,4,7)
ax2.plot(range(7),df_auc.ix["l2_reg",],"r^-")
plt.xlabel("L2 Regularization\n(g)")
plt.ylabel("AUC")
plt.xlim((0,6))
plt.xticks(range(7),["1e-7","1e-6","1e-5","1e-4","1e-3","1e-2","1e-1"])
plt.grid(linestyle='-.')

# L2 Regularization
ax1=fig.add_subplot(2,4,4)
ax1.plot(range(7),df_loss.ix["dropout",],"bv-")
plt.xlabel("Dropout Rate\n(d)")
plt.ylabel("Logloss")
plt.xlim((0,6))
plt.xticks(range(7),["0.1","0.2","0.3","0.4","0.5","0.6","0.7"])
plt.grid(linestyle='-.')

ax2=fig.add_subplot(2,4,8)
ax2.plot(range(7),df_auc.ix["dropout",],"r^-")
plt.xlabel("Dropout Rate\n(h)")
plt.ylabel("AUC")
plt.xlim((0,6))
plt.xticks(range(7),["0.1","0.2","0.3","0.4","0.5","0.6","0.7"])
plt.grid(linestyle='-.')

plt.tight_layout() #设置默认的间距
plt.savefig('impact_of_parameters.png', dpi=1000) #指定分辨率保存
plt.show()
