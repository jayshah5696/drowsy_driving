# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:01:21 2018

@author: Lenovo
"""


train_all =pd.read_csv("E:\\USA\\Projects\\Research\\R_code\w5\\weights\\train_all.csv")
train_all=train_all[['RunName','win60s','drowsy']]

index_all=pd.read_csv("E:\\USA\\Projects\\Research\\R_code\w6\\index_all.csv")
index_all=index_all.drop(index_all.columns[[0]],axis=1)


train_all['new']= train_all['win60s'].astype(str)+ train_all['RunName']
index_all['new']=index_all['win60s'].astype(str)+ index_all['RunName']
    

index_all=train_all.merge(index_all[['c','new']],how='left')


prob=pd.DataFrame(yhat) 
index_all.shape
prob.shape
#Plotting error graph


combine = pd.concat([index_all,prob],axis=1)
combine.head()
combine.to_csv("E:\\USA\\Projects\\Research\\R_code\\w6\\combine_plot.csv")


combine1 = combine[['c', 0,1]]
c0=combine1.groupby('c')[0].mean()
c1=combine1.groupby('c')[1].mean()
c2=combine1.groupby('c')[0].max()
c3=combine1.groupby('c')[0].min()
c4=combine1.groupby('c')[1].max()
c5=combine1.groupby('c')[1].min()
c6=combine1.groupby('c')['c'].count()


ind=[c6>10]
c2=c2-c3
c4=c4-c5

final=pd.concat([c0,c1,c2,c4],axis=1)
final.columns=['mean0','mean1','range0','range1']
final.columns
final.head()

a=pd.DataFrame(final.index)
type(a)
a.head()
a['c']=a['c'].astype('category')

import matplotlib.pyplot as plt
x=a.c
y=final.mean0
err=final.range0
plt.errorbar(x,y,yerr=err,capsize=1,fmt='o')
plt.show



conbine2=combine1[c6>10]
only_0=combine1[~ytrain_all]
cpivot=combine1.pivot(columns='c',values=1)

cpivot=combine1.pivot(columns='c',values=1)
cpivot.columns
data=cpivot[[101.0,102.0,103.0,104.0,105.0,106.0,111.0,201.0,202.0,203.0,204.0,205.0,206.0,301.0,302.0,303.0,304.0,305.0,306.,307.0,308.0]]
data.columns=["101_1","102_1","103_1","104_1","105_1","106_1","111_1","201_1","202_1","203_1","204_1","205_1","206_1","301_1","302_1","303_1","304_1","305_1","306_1","307_1","308_1"]
data.to_csv("E:\\USA\\Projects\\Research\\R_code\\w6\\pivot_plot.csv")


plt.boxplot(c0)
plt.show

j=cpivot.columns[[0]]
d1=cpivot.iloc[:,[0]].dropna()
plt.boxplot(d1)



ytrain_all=train_all.iloc[:,3]
combine1[ytrain_all]