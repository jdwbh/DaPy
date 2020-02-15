# -*- coding: utf-8 -*-
"""
###   Python数据分析基础教程
###    函数库 DaPy_fun.py
###   王斌会 王术 2018-6-1
（1）安装库：将DaPy_fun.py文档拷贝到当前工作目录下
（2）调用包：from DaPy_fun import *
（3）使用函数：tab(x)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

def tab(x,plot=False): #计数频数表和绘图
   f=x.value_counts();f
   s=sum(f);
   p=round(f/s*100,3);p
   T1=pd.concat([f,p],axis=1);
   T1.columns=['例数','构成比'];
   T2=pd.DataFrame({'例数':s,'构成比':100.00},index=['合计'])
   Tab=T1.append(T2)
   if plot:
     fig,ax = plt.subplots(1,2,figsize=(15,6))
     ax[0].bar(f.index,f); # 条图
     ax[1].pie(p,labels=p.index);  # 饼图
   return(round(Tab,3))

def freq(X,bins=10,density=False): #计量频数表与直方图
    if density:
        H=plt.hist(X,bins,density=density)
        plt.plot(H[1],st.norm.pdf(H[1]),color='r');
    else:
       H=plt.hist(X,bins);
    a=H[1][:-1];a
    b=H[1][1:];b
    f=H[0];f
    p=f/sum(f)*100;p
    cp=np.cumsum(p);cp
    Freq=pd.DataFrame([a,b,f,p,cp],index=['[下限','上限)','频数','频率(%)','累计频数(%)'])
    return(round(Freq.T,2))

def stats(x): #基本统计量
    stat=[x.count(),x.min(),x.quantile(.25),x.mean(),x.median(),
           x.quantile(.75),x.max(),x.max()-x.min(),x.var(),x.std(),x.skew(),x.kurt()]
    stat=pd.Series(stat,index=['Count','Min', 'Q1(25%)','Mean','Median',
                               'Q3(75%)','Max','Range','Var','Std','Skew','Kurt'])
    return(stat)

def norm_p(a=-2,b=2): ### 正态曲线面积图
    x=np.arange(-4,4,0.1)
    y=st.norm.pdf(x)
    x1=x[(a<=x) & (x<=b)];x1
    y1=y[(a<=x) & (x<=b)];y1
    p=st.norm.cdf(b)-st.norm.cdf(a);p
    #plt.title("N(0,1)分布: [%6.3f %6.3f] p=%6.4f"%(a,b,p))
    plt.plot(x,y);
    plt.hlines(0,-4,4);
    plt.vlines(x1,0,y1,colors='r');
    plt.text(-0.5,0.2,"%5.2f％" % (p*100.0),fontsize=15);

#norm_p(1.6,4)
#norm_p(1.24,4)
#norm_p(-0.67,4)

def t_p(a=-2,b=2,df=10,k=0.1):
    x=np.arange(-4,4,k)
    y=st.t.pdf(x,df)
    x1=x[(a<=x) & (x<=b)];x1
    y1=y[(a<=x) & (x<=b)];y1
    p=st.t.cdf(b,df)-st.t.cdf(a,df);p
    plt.plot(x,y);
    plt.title("t(%2d): [%6.3f %6.3f] p=%6.4f"%(df,a,b,p))
    plt.hlines(0,-4,4); plt.vlines(x1,0,y1,colors='r');
    plt.text(-0.5,0.2,"p=%6.4f" % p,fontsize=15);

from math import sqrt
def t_interval(b,x):
    a=1-b
    n = len(x)
    ta=st.t.ppf(1-a/2,n-1);ta
    se=x.std()/sqrt(n)
    return(x.mean()-ta*se, x.mean()+se*ta)
#t_interval(0.95,BSdata['身高'])

def ttest_1plot(X,mu=0): # 单样本均值t检验图
    k=0.1
    df=len(X)-1
    t1p=st.ttest_1samp(X, popmean = mu);
    x=np.arange(-4,4,k); y=st.t.pdf(x,df)
    t=abs(t1p[0]);p=t1p[1]
    x1=x[x<=-t]; y1=y[x<=-t];
    x2=x[x>=t]; y2=y[x>=t];
    print("  单样本t检验\t t=%6.3f p=%6.4f"%(t,p))
    print("  t置信区间: ",st.t.interval(0.95,len(X)-1,X.mean(),X.std()))
    plt.plot(x,y); plt.hlines(0,-4,4);
    plt.vlines(x1,0,y1,colors='r'); plt.vlines(x2,0,y2,colors='r');
    plt.text(-0.5,0.05,"p=%6.4f" % t1p[1],fontsize=15);
    plt.vlines(st.t.ppf(0.05/2,df),0,0.2,colors='b');
    plt.vlines(-st.t.ppf(0.05/2,df),0,0.2,colors='b');
    plt.text(-0.5,0.2,r"$\alpha$=%3.2f"%0.05,fontsize=15);

def reglinedemo(n=20):    #模拟直线回归
    x=np.arange(n)+1
    e=np.random.normal(0,1,n)
    y=2+0.5*x+e
    import statsmodels.api as sm
    x1=sm.add_constant(x);x1
    fm=sm.OLS(y,x1).fit();fm
    plt.plot(x,y,'.',x,fm.fittedvalues,'r-'); #添加回归线，红色
    for i in range(len(x)):
        plt.vlines(x,y,fm.fittedvalues,linestyles='dotted',colors='b');
#reglinedemo(30);   #最小二乘回归示意图---直线回归
