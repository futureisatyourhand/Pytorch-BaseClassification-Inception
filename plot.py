# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 11:50:49 2020

@author: hlj0812.duapp.com
"""

from matplotlib import pyplot as plt
import numpy as np
files=open('case1.txt','r')
case1=files.readlines()
files.close()

files=open('case2.txt','r')
case2=files.readlines()
files.close()

files=open('case3.txt','r')
case3=files.readlines()
files.close()

files=open('case4.txt','r')
case4=files.readlines()
files.close()

content=[]
i=0
for c in case1:
    if i%200==0:
        content.append(list(map(float,c.strip('\n').split(' ')[1:])))
    i+=1
case1=np.array(content)

content=[]
i=0
for c in case2:
    if i%200==0:
        content.append(list(map(float,c.strip('\n').split(' ')[1:])))
    i+=1
case2=np.array(content)

content=[]
i=0
for c in case3:
    if i%200==0:
        content.append(list(map(float,c.strip('\n').split(' ')[1:])))
    i+=1
case3=np.array(content)

content=[]
i=0
for c in case4:
    if i%200==0:
        content.append(list(map(float,c.strip('\n').split(' ')[1:])))
    i+=1
case4=np.array(content)

x=list(range(1,case1.shape[0]+1))
loss1,=plt.plot(x,case1[:,0],color='blue',linewidth = '2')
loss2,=plt.plot(x,case2[:,0],color='darkorange',linewidth = '2')
loss3,=plt.plot(x,case3[:,0],color='green',linewidth = '2')
loss4,=plt.plot(x,case4[:,0],color='red',linewidth = '2')
plt.legend(handles = [loss1,loss2,loss3,loss4], 
           labels = ['case1','case2','case3','case4'],
           loc='upper right')
plt.title('losses of different poolings')
plt.xlabel('iteration*200')
plt.ylabel('losses')
plt.show()
