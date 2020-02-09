## -*- coding: utf-8 -*-
"""
###   Python数据分析基础教程（DaPy）
###   代码档 DaPy_code.py
###   函数库 DaPy_fun.py
###   数据框 DaPy_data.xlsx
###   王斌会 王术 2019-6-19
"""

### 初始化
%run init.py
#import DaPy_fun *

### 读数据csv（本地）
#BSdata=pd.read_csv('BSdata.csv',encoding='utf-8'); BSdata[:6]
### 读数据csv（云端）
#url1='http://leanote.com/api/file/getAttach?fileId=5abbb388ab6441507e002161'
#dat1=pd.read_csv(url1,encoding='utf-8'); dat1

### 读数据xlsx（本地）
import pandas as pd                        #加载数据分析包
BSdata=pd.read_excel('DaPy_data.xlsx','BSdata'); BSdata[:6]
### 读数据csv（云端）
#url2='http://leanote.com/api/file/getAttach?fileId=5abbb3aaab6441507e002167'
#dat2=pd.read_excel(url2,'BSdata'); dat2

#
#第1章 数据收集与软件应用
##1.3 Python编程基础
#### 1.3.1.1 Python的工作目录
'''获得当前目录'''
pwd
'''改变工作目录'''
cd "D:\\DaPy1"
pwd

!dir

#3 Python 编程分析基础
##3.1 Python 数据类型
###3.1.1Pyhton 对象
who
x=10.12   #创建对象x
who
del x   #删除对象x
who

###3.1.2 数据基本类型
#数值型
n=10       #整数
n
print("n=",n)
x=10.234   #实数
print(x)
print("x=%10.5f"%x)

#逻辑型
a=True;a
b=False;b

10>3
10<3

#字符型
s='IlovePython';s
s[7]
s[2:6]
s+s
s*2

float('nan')

###3.1.3 标准数据类型
#（1）List（列表）
list1=[] # 空列表
list1
list1=['Python',786,2.23,'R',70.2]
list1 # 输出完整列表
list1[0] # 输出列表的第一个元素
list1[1:3] # 输出第二个至第三个元素
list1[2:] # 输出从第三个开始至列表末尾的所有元素
list1*2 # 输出列表两次
list1+list1[2:4] # 打印组合的列表

X=[1,3,6,4,9];X
sex=[' 女',' 男',' 男',' 女',' 男']
sex
weight=[67,66,83,68,70];
weight

#（2）Tuple（元组）

#（3）Dictionary（字典）
{}            #空字典
dict1={'name':'john','code':6734,'dept':'sales'};dict1 #定义字典
dict1['code']  # 输出键为'code' 的值
dict1.keys()   # 输出所有键
dict1.values() # 输出所有值

dict2={'sex': sex,'weight':weight}; dict2 #根据列表构成字典

##3.2 数值分析库numpy
###3.2.1 一维数组(向量)
import numpy as np       #加载数组包
np.array([1,2,3,4,5])         #一维数组
np.array([1,2,3,np.nan,5])    #包含缺失值的数组

np.array(X)              #列表变数组
np.arange(9)             #数组序列
np.arange(1,9,0.5)       #等差数列
np.linspace(1,9,5)       #等距数列

np.random.randint(1,9)   #1~9随机数
np.random.rand(10)       #10个均匀随机数
np.random.randn(10)      #10个正态随机数

###3.2.2 二维数组(矩阵)
np.array([[1,2],[3,4],[5,6]]) #二维数组
A=np.arange(9).reshape((3,3));A # 形成3x3

###3.2.3 数组的操作
A.shape
np.empty([3,3]) #空数组
np.zeros((3,3)) #零矩阵
np.ones((3,3))  #1矩阵
np.eye(3)       #单位阵

##3.3 数据分析库pandas
import pandas as pd   #加载数据分析包

#书37页两个（1）（1）

###3.3.1 序列:Seriers
#（1）创建序列（向量、一维数组）
pd.Series()           #生成空序列

#（2）根据列表构建序列
X=[1,3,6,4,9]
S1=pd.Series(X);S1
S2=pd.Series(weight);S2
S3=pd.Series(sex);S3

#（3）序列合并
pd.concat([S2,S3],axis=0)    #按行并序列
pd.concat([S2,S3],axis=1)    #按列并序列

#（4）序列切边
S1[2]
S3[1:4]

###3.3.2 数据框:DataFrame
#（1）生成数据框
pd.DataFrame()      #生成空数据框

#（2）根据列表创建数据框
pd.DataFrame(X)
pd.DataFrame(X,columns=['X'],index=range(5))
pd.DataFrame(weight,columns=['weight'],index=['A','B','C','D','E'])

#（3）根据字典创建数据框
'''通过字典列表生成数据框是Python较快捷的方式 '''
df1=pd.DataFrame({'S1':S1,'S2':S2,'S3':S3});df1
df2=pd.DataFrame({'sex':sex,'weight':weight},index=X);df2

#（4）增加数据框列
df2['weight2']=df2.weight**2; df2   # 生成新列

#（5）删除数据框列
del df2['weight2']; df2   #删除数据列

#（5）缺失值处理
df3=pd.DataFrame({'S2':S2,'S3':S3},index=S1);df3
df3.isnull()#是缺失值返回True，否则范围False
df3.isnull().sum()#返回每列包含的缺失值的个数
df3.dropna()   #直接删除含有缺失值的行，多变量谨慎使用
#df3.dropna(how = 'all')#只删除全是缺失值的行

#（7）数据框排序
df3.sort_index()         #按index排序
df3.sort_values(by='S3') #按列值排序

###3.3.3 数据框的读写
####3.3.3.1pandas读取数据集
#（1）从剪切板上读取
#BSdata=pd.read_clipboard();
BSdata[:5]  #从剪切板上复制数据

#（2）读取csv格式数据
#BSdata=pd.read_csv("BSdata.csv",encoding='utf-8') #注意中文格式
BSdata[6:9]

#（3）读取Excel格式数据
BSdata=pd.read_excel('DaPy_data.xlsx','BSdata');BSdata[-5:]

####3.3.3.2pandas数据集的保存
BSdata.to_csv('BSdata1.csv') #将数据框BSdata保存到BSdata.csv

###3.3.4 数据框的操作
####3.3.4.1 基本信息

#（1）数据框显示
BSdata.info()            #数据框信息
BSdata.head()            #显示前5行
BSdata.tail()            #显示后5行

#（2）数据框列名（变量名）
BSdata.columns           #查看列名称

#（3）数据框行名（样品名）
BSdata.index             #数据框行名

#（4）数据框维度
BSdata.shape             #显示数据框的行数和列数
BSdata.shape[0]          #数据框行数
BSdata.shape[1]          #数据框列数

#（5）数据框值（数组）
BSdata.values            #数据框值数组

####3.3.4.2 选取变量
BSdata.身高 # 取一列数据，BSdata['身高']
BSdata[['身高','体重']]  #取两列数据
BSdata.iloc[:,2] # 取1列
BSdata.iloc[:,2:4] # 取3 、4 列

####3.3.4.3 提取样品
BSdata.loc[3] #取1行
BSdata.loc[3:5] #取3-5行

####3.3.4.4 选取观测与变量
BSdata.loc[:3,['身高','体重']]
BSdata.iloc[:3,:5] #0到2行和1:5

####3.3.4.5 条件选取
BSdata[BSdata['身高']>180]
BSdata[(BSdata['身高']>180)&(BSdata['体重']<80)]

####3.3.4.6 数据框的运算
BSdata['体重指数']=BSdata['体重']/(BSdata['身高']/100)**2
round(BSdata[:5],2)

pd.concat([BSdata.身高, BSdata.体重],axis=0)
pd.concat([BSdata.身高, BSdata.体重],axis=1)

BSdata.iloc[:3,:5].T

#3.4 Python 编程运算
##3.4.1 基本运算
##3.4.2 控制语句
####3.4.2.1 循环语句for
for i in range(1,5):
    print(i)

fruits = ['banana', 'apple',  'mango']
for fruit in fruits:
   print('当前水果 :', fruit)

for var in BSdata.columns:
    print(var)

####3.4.2.2 条件语句if/else
a = -100
if a < 100:
    print("数值小于100")
else:
    print("数值大于100")

-a if a<0 else a

##3.4.3 函数定义

x=[1,3,6,4,9,7,5,8,2];x
def xbar(x):
    n=len(x)
    xm=sum(x)/n
    return(xm)
xbar(x)
np.mean(x)

##3.4.4 面向对象
def SS1(x):
    n=len(x)
    ss=sum(x**2)-sum(x)**2/n
    return(ss)
SS1(X) #SS1(BSdata. 身高)

def SS2(x): # 返回多个值
    n=len(x)
    xm=sum(x)/n
    ss=sum(x**2)-sum(x)**2/n
    return[x**2,n,xm,ss]#return(x**2,n,xm,ss)
SS2(X) #SS2(BSdata.身高)

SS2(X)[0] # 取第1 个对象
SS2(X)[1] # 取第2 个对象
SS2(X)[2] # 取第3 个对象
SS2(X)[3] # 取第4 个对象

type(SS2(X))
type(SS2(X)[3])
#数据及练习1

#4 数据的探索性分析
##4.1 数据的描述分析
###4.1.1基本描述统计量
BSdata.describe()
BSdata[['性别','开设','课程','软件']].describe()

####4.1.2计数数据汇总分析
#（1）频数：绝对数
T1=BSdata.性别.value_counts();T1

#（2）频率：相对数
T1/sum(T1)*100

###4.1.3 计量数据汇总分析
#（1）均数（算术平均数）
BSdata.身高.mean()
#（2）中位数
BSdata.身高.median()
#（3）极差
BSdata.身高.max()-BSdata.身高.min()
#（4）方差
BSdata.身高.var()
#（5）标准差
BSdata.身高.std()
#（6）四分位数间距
BSdata.身高.quantile(0.75)-BSdata.身高.quantile(0.25)
#（7）偏度
BSdata.身高.skew()
#（8）峰度
BSdata.身高.kurt()

#（9）自定义计算基本统计量函数
def stats(x):
    stat=[x.count(),x.min(),x.quantile(.25),x.mean(),x.median(),
         x.quantile(.75),x.max(),x.max()-x.min(),x.var(),x.std(),x.skew(),x.kurt()]
    stat=pd.Series(stat,index=['Count','Min', 'Q1(25%)','Mean','Median',
                   'Q3(75%)','Max','Range','Var','Std','Skew','Kurt'])
    x.plot(kind='kde') #拟合核密度 kde 曲线，见下节
    return(stat)
stats(BSdata.身高)
stats(BSdata.支出)

#4.2 基本绘图命令
##4.2.1 常用的绘图函数
import DaPy1func as da
da.stats(BSdata.身高)
da.stats(BSdata.支出)

import matplotlib.pyplot as plt              #基本绘图包
plt.rcParams['font.sans-serif']=['KaiTi'];   #SimHei黑体
plt.rcParams['axes.unicode_minus']=False;    #正常显示图中负号
plt.figure(figsize=(6,5));                   #图形大小
'''本地直接显示图形'''
matplotlib inline

#（1）常用的统计图函数
#（2）图形参数设置
####二、计数数据的基本统计图
X=['A','B','C','D','E','F','G']
Y=[1,4,7,3,2,5,6]
plt.bar(X,Y); # 条图
plt.pie(Y,labels=X);  # 饼图
#plt.pie(Y,labels=X,autopct='%1.2f%%')

plt.plot(X,Y)  #线图 plot

plt.hist(BSdata.身高)  # 频数直方图
plt.hist(BSdata.身高,density=True) # 频率直方图

plt.scatter(BSdata.身高, BSdata.体重);  # 散点图
plt.xlabel(u'身高');plt.ylabel(u'体重');

#（3）图形参数设置
plt.plot(X,Y,c='red');
plt.ylim(0,8);
plt.xlabel('names');plt.ylabel('values');
plt.xticks(range(len(X)), X);
plt.plot(X,Y,linestyle='--',marker='o');

plt.plot(X,Y,'o--'); plt.axvline(x=1);plt.axhline(y=4);
#plt.vlines(1,0,6,colors='r');plt.hlines(4,0,6);

plt.plot(X,Y);plt.text(2,7,'peakpoint')
plt.plot(X,Y,label=u'折线');
plt.legend();

#误差条图
s=[0.1,0.4,0.7,0.3,0.2,0.5,0.6]
plt.bar(X,Y,yerr=s,error_kw={'capsize':5})

#（4）多图
plt.figure(figsize=(12,6));
plt.subplot(121); plt.bar(X,Y);
plt.subplot(122); plt.plot(Y);

plt.figure(figsize=(7,10));
plt.subplot(211); plt.bar(X,Y);
plt.subplot(212); plt.plot(Y);

fig,ax = plt.subplots(1,2,figsize=(14,6))
ax[0].bar(X,Y)
ax[1].plot(X,Y)

fig,ax=plt.subplots(2,2,figsize=(15,10))
ax[0,0].bar(X,Y); ax[0,1].pie(Y,labels=X)
ax[1,0].plot(Y); ax[1,1].plot(Y,'.-',linewidth=3);

###4.2.2 基于pandas 的绘图
BSdata['体重'].plot(kind='line');
BSdata['体重'].plot(kind='hist')
BSdata['体重'].plot(kind='box');
BSdata['体重'].plot(kind='density',title='Density');

BSdata[['身高','体重','支出']].plot(subplots=True,layout=(1,3),kind='box')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(1,3),kind='density')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(3,1),kind='density')


T1=BSdata['开设'].value_counts();T1
pd.DataFrame({'频数':T1,'频率':T1/T1.sum()*100})
T1.plot(kind='bar'); #T1.sort_values().plot(kind='bar');
T1.plot(kind='pie');

##4.3 数据的分类分析
###4.3.1 一维频数分析
####4.3.1.1 计数数据频数分布
#（1）pivot_table
BSdata['开设'].value_counts()
#pd.pivot_table(BSdata,values='学号',index='开设',aggfunc=len)
#BSdata.pivot_table(values='学号',index='开设',aggfunc=len)
# (2) 计数频数表
def tab(x,plot=False): #计数频数表
   f=x.value_counts();f
   s=sum(f);
   p=round(f/s*100,3);p
   T1=pd.concat([f,p],axis=1);
   T1.columns=['例数','构成比'];
   T2=pd.DataFrame({'例数':s,'构成比':100.00},index=['合计'])
   Tab=T1.append(T2)
   if plot:
     fig,ax = plt.subplots(2,1,figsize=(8,15))
     ax[0].bar(f.index,f); # 条图
     ax[1].pie(p,labels=p.index,autopct='%1.2f%%');  # 饼图
   return(round(Tab,3))

tab(BSdata.开设,True)


####4.3.1.2 计量数据频数分布
#（1）身高频数表
pd.cut(BSdata.身高,bins=10).value_counts()
pd.cut(BSdata.身高,bins=10).value_counts().plot(kind='bar');
#（2）支出频数表
pd.cut(BSdata.支出,bins=[0,10,30,100]).value_counts()
pd.cut(BSdata.支出,bins=[0,10,30,100]).value_counts().plot(kind='bar');

# (3) 计量频数表
def freq(X,bins=10): #计量频数表与直方图
    H=plt.hist(X,bins);
    a=H[1][:-1];a
    b=H[1][1:];b
    f=H[0];f
    p=f/sum(f)*100;p
    cp=np.cumsum(p);cp
    Freq=pd.DataFrame([a,b,f,p,cp])
    Freq.index=['[下限','上限)','频数','频率(%)','累计频数(%)']
    return(round(Freq.T,2))

freq(BSdata.体重)


###4.3.2 二维集聚分析
####4.3.2.1 计数数据的列联表
#（1）二维列联表
pd.crosstab(BSdata.开设,BSdata.课程)
pd.crosstab(BSdata.开设,BSdata.课程,margins=True)

pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='index')
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='columns')
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='all').round(3)

BSdata.pivot_table('学号','开设','课程',aggfunc=len)
BSdata.pivot_table('学号',index='开设',columns='课程',aggfunc=len)
pd.pivot_table(BSdata,values='学号',index='开设',columns='课程',aggfunc=len)

BSdata
#（2）复式条图
T2=pd.crosstab(BSdata.开设,BSdata.课程);T2
T2.plot(kind='bar');

T2.plot(kind='barh');
T2.plot(kind='bar',stacked=True);

####4.3.2.2 计量数据的集聚表
#（1）groupby函数
BSdata.groupby(['性别'])
type(BSdata.groupby(['性别']))

BSdata.groupby(['性别'])['身高'].mean()
BSdata.groupby(['性别'])['身高'].size()
BSdata.groupby(['性别','开设'])['身高'].mean()

#（2）agg函数
BSdata.groupby(['性别'])['身高'].agg([np.mean, np.std])

#（3）应用apply()
BSdata.groupby(['性别'])['身高','体重'].apply(np.mean)
BSdata.groupby(['性别','开设'])['身高','体重'].apply(np.mean)


###4.3.3 多维透视分析
####4.2.3.1 计数数据的透视分析
#（1）pivot_table
BSdata.pivot_table(index=['性别'],values=['学号'],aggfunc=len)
BSdata.pivot_table(values=['学号'],index=['性别','开设'],aggfunc=len)
BSdata.pivot_table(values=['学号'],index=['开设'],columns=['性别'],aggfunc=len)

####4.2.3.2 计量数据的透视分析
#pd.pivot_table(BSdata,index=["性别"],aggfunc=len)
BSdata.pivot_table(index=['性别'],values=["身高"],aggfunc=np.mean)
BSdata.pivot_table(index=['性别'],values=["身高"],aggfunc=[np.mean,np.std])
BSdata.pivot_table(index=["性别"],values=["身高","体重"])

####4.2.3.3 复合数据的透视分析
pd.pivot_table(BSdata,index=["性别","开设"],aggfunc=len,margins=True)
BSdata.pivot_table('学号', ['性别','开设'], '课程', aggfunc=len, margins=True, margins_name='合计')
pd.pivot_table(BSdata,index=["性别"],aggfunc=np.mean)
BSdata.pivot_table(['身高','体重'],['性别',"开设"],aggfunc=[len,np.mean,np.std] )

#5 数据的可视化分析
##5.1 特殊统计图绘制
###5.1.1 数学函数图
#（1）初等函数
import math
x=np.linspace(0,2*math.pi);x  #[0,2*pi]序列
#fig,ax=plt.subplots(2,2,figsize=(15,12))
plt.plot(x,np.sin(x))
plt.plot(x,np.cos(x))
plt.plot(x,np.log(x))
plt.plot(x,np.exp(x))
#（2）极坐标图	（加公式）
t=np.linspace(0,2*math.pi)
x=2*np.sin(t)
y=3*np.cos(t)
plt.plot(x,y)
plt.text(0,0,r'$\frac{x^2}{2}+\frac{y^2}{3}=1$',fontsize=15)
#（3）三维曲面图
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
X=np.linspace(-4,4,20) #X = np.arange(-4, 4, 0.5);
Y=np.linspace(-4,4,20) #Y = np.arange(-4, 4, 0.5)
X, Y = np.meshgrid(X, Y)
Z = np.sqrt(X**2 + Y**2)
ax.plot_surface(X, Y, Z);

#气泡图
plt.scatter(BSdata['身高'], BSdata['体重'], s=BSdata['支出']);
#三维三点图
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(BSdata['身高'], BSdata['体重'], BSdata['支出'])

#2.3.1.6 统计地图
#pip3 install https://github.com/matplotlib/basemap/archive/v1.1.0.tar.gz
#from mpl_toolkits.basemap import Basemap
#plt.figure(figsize=(16,8))
#m = Basemap()
#m.drawcoastlines()

##5.2 Seaborn统计绘图
import seaborn as sns

#（1）箱线图boxplot
# 绘制箱线图
sns.boxplot(x=BSdata['身高'])
# 竖着放的箱线图，也就是将x换成y
sns.boxplot(y=BSdata['身高'])
# 分组绘制箱线图，分组因子是性别，在x轴不同位置绘制
sns.boxplot(x='性别', y='身高',data=BSdata)
# 分组箱线图，分子因子是smoker，不同的因子用不同颜色区分, 相当于分组之后又分组
sns.boxplot(x='开设', y='支出',hue='性别',data=BSdata)

#（2）小提琴图violinplot
sns.violinplot(x='性别', y='身高',data=BSdata)
sns.violinplot(x='开设', y='支出',hue='性别',data=BSdata)

#（3）散点图striplot
sns.stripplot(x='性别', y='身高',data=BSdata)
sns.stripplot(x='性别', y='身高',data=BSdata,jitter=True)
sns.stripplot(y='性别', x='身高',data=BSdata,jitter=True)

#（4）条图barplot
sns.barplot(x='性别', y='身高',data=BSdata,ci=0,palette="Blues_d")

#（5）计数的直方图countplot
# 分组绘图
sns.countplot(x='性别',data=BSdata)
sns.countplot(y='开设',data=BSdata)
sns.countplot(x='性别',hue="开设",data=BSdata)

#（6）两变量关系图factorplot
# 不同的deck（因子）绘制不同的alive（数值），col为分子图绘制，col_wrap每行画4个子图
sns.factorplot(x='性别',col="开设", col_wrap=3,data=BSdata, kind="count", size=2.5, aspect=.8)
#（7）概率分布图
sns.distplot(BSdata['身高'], kde=True, bins=20, rug=True);

sns.jointplot(x='身高', y='体重', data=BSdata);

sns.pairplot(BSdata[['身高','体重','支出']]);

##5.3 ggplot绘图系统
from plotnine import *    #加载和调用ggplot所有方法
#（1）直方图
qplot('身高',data=BSdata, geom='histogram')
#（2）条形图
qplot('开设',data=BSdata, geom='bar')
#（3）散点图
qplot('身高','体重',data=BSdata,color='性别');
qplot('身高','体重',data=BSdata,color='性别',size='性别');

###5.3.2 ggplot基本绘图
#（2）图层的概念
GP=ggplot(aes(x='身高',y='体重'),data=BSdata);GP #绘制直角坐标系
GP + geom_point()                  #增加点图
GP + geom_line()                   #增加线图

ggplot(BSdata,aes(x='身高',y='体重')) + geom_point() + geom_line()
ggplot(BSdata,aes(x='身高',y='体重',color='性别'))+geom_point()+theme_bw()

#（4）常见统计图
ggplot(BSdata,aes(x='身高'))+ geom_histogram()

ggplot(BSdata,aes(x='支出'))+geom_line(aes(y='身高'))
ggplot(BSdata,aes(x='支出'))+geom_line(aes(y='身高'))+geom_line(aes(y='体重'))
ggplot(BSdata,aes(x='身高')) + geom_histogram() + facet_wrap('性别')

ggplot(BSdata,aes(x='身高',y='体重')) + geom_point()
ggplot(BSdata,aes(x='身高',y='体重',color='性别'))+geom_point()
ggplot(BSdata,aes(x='身高',y='体重',color='性别'))+geom_point()+theme_bw()

ggplot(BSdata,aes(x='身高',y='体重',size='开设',colour='性别'))+geom_point()
ggplot(BSdata,aes(x='身高',colour='性别',fill='True')) + geom_density()
ggplot(BSdata,aes(x='身高',y='体重')) + geom_point() + facet_wrap('性别')

ggplot(BSdata,aes(x='身高',y='体重',color='性别'))+geom_point()+theme_bw()

### pyecharts
from pyecharts import Bar
bar = Bar('柱形图')
df = pd.DataFrame({'x':[1,2,3], 'y':[4,2,6]})
bar.add('first add', df.x, df.y)
bar

import pandas as pd
import numpy as np
df = pd.DataFrame([np.random.uniform(10,1,size=1000),
                   np.random.uniform(10,5,size=1000),
                   np.random.randint(1,high=10,size=1000),
                   np.random.choice(list('ABCD'),size=1000)],
                  index=['col1','col2','col3','col4']).T
df

from eplot import eplot
df.eplot()

df = pd.Series([4,2,6],index=[1,2,3])
df.eplot.bar(title='柱形图')


#6 数据的统计分析
##6.1 随机变量及其分布
####6.1.1 均匀分布
a=0;b=1;y=1/(b-a)
plt.plot(a,y); plt.hlines(y,a,b);
plt.show()
#plt.vlines(0,0,1);plt.vlines(1,0,1);

#####（1）整数随机数
import random
random.randint(10,20)  #[10,20]上的随机整数

#####（2）实数随机数
random.uniform(0,1)    #[0,1]上的随机实数

#####（3）整数随机数列
import numpy as np
np.random.randint(10,21,9)  #[10,20]上的随机整数

#####（4）实数随机数列
np.random.uniform(0,1,10)   #[0,1]上的10个随机实数=np.random.rand(10)

###6.1.2 正态分布
#####（2）标准正态分布
from math import sqrt,pi   #调用数学函数，import math as *
x=np.linspace(-4,4,50);
y=1/sqrt(2*pi)*np.exp(-x**2/2);
plt.plot(x,y);
plt.show()

import scipy.stats as st  #加载统计方法包
P=st.norm.cdf(2);P

'''加载自定义库,在当前目录下建立DaPy1func.py函数库即可'''
import DaPy_fun as da
'''标准正态曲线面积（概率） '''
da.norm_p(-1,1)         #68.27%
da.norm_p(-2,2)         #94.45%
da.norm_p(-1.96,1.96)   #95%
da.norm_p(-3,3)         #99.73%
da.norm_p(-2.58,2.58)   #99%

za=st.norm.ppf(0.95);za   #单侧
[st.norm.ppf(0.025),st.norm.ppf(0.975)]  #双侧

#####（3）正态随机数
np.random.normal(10,4,5)  #产生5个均值为10标准差为4的正态随机数
np.random.normal(0,1,5)   #生成5个标准正态分布随机数

'''一页绘制四个正态随机图 '''
fig,ax = plt.subplots(2,2)
for i in range(2):
    for j in range(2):
        ax[i,j].hist(np.random.normal(0,1,500),bins = 50)
plt.subplots_adjust(wspace = 0,hspace=0)

z=np.random.normal(0,1,100)
#cnts, bins = np.histogram(z, bins=50, normed=True)
#bins = (bins[:-1] + bins[1:]) / 2
#plt.hist(z,bins=50,density=True)
#plt.plot(bins, cnts)
#plt.hist(z,density=True)[0]
#plt.hist(z,density=True)[1]
import seaborn as sns
sns.distplot(z)

st.probplot(BSdata.身高, dist="norm", plot=plt); #正态概率图
st.probplot(BSdata['支出'], dist="norm", plot=plt);

##6.2 数据分析统计基础
###6.2.1 的 统计量的
#####（1）简单随机抽样
np.random.randint(0,2,10)  #[0,2)上的10个随机整数
i=np.random.randint(1,53,6);i #抽取10个学生，[1,52]上的6个整数
BSdata.iloc[i]       #随机抽取的6个学生信息
BSdata.sample(6)    #直接抽取6个学生的信息

###6.2.2 统计量的分布
def norm_sim1(N=1000,n=10):    # n样本个数, N模拟次数（即抽样次数）
    xbar=np.zeros(N)            #模拟样本均值
    for i in range(N):          #[0,1]上的标准正态随机数及均值
       xbar[i]=np.random.normal(0,1,n).mean()
    sns.distplot(xbar,bins=50)  #plt.hist(xbar,bins=50)
    print(pd.DataFrame(xbar).describe().T)
norm_sim1()
norm_sim1(10000,30)
#sns.distplot(norm_sim1())  #plt.hist(norm_sim1())
#sns.distplot(norm_sim1(n=30,N=10000)) #plt.hist(norm_sim1(n=30,N=10000))

def norm_sim2(N=1000,n=10):
    xbar=np.zeros(N)
    for i in range(N):
       xbar[i]=np.random.uniform(0,1,n).mean()  #[0,1]上的均匀随机数及均值
    sns.distplot(xbar,bins=50)
    print(pd.DataFrame(xbar).describe().T)
norm_sim2()
norm_sim2(10000,30)

#sns.distplot(norm_sim2())             #plt.hist(norm_sim2())
#sns.distplot(norm_sim1(n=30,N=10000)) #plt.hist(norm_sim2(n=30,N=10000))

#####（3）t分布曲线
x=np.arange(-4,4,0.1)
yn=st.norm.pdf(x,0,1);yt3=st.t.pdf(x,3);yt10=st.t.pdf(x,10)
plt.plot(x,yn,'r-',x,yt3,'b.',x,yt10,'g-.');
plt.legend(["N(0,1)","t(3)","t(10)"]);

##6.3  基本统计推断方法

###3.2.1 参数的估计方法
####3.2.1.1 点估计
#####（1）均值的点估计
BSdata['身高'].mean()
#####（2）标准差的点估计
BSdata['身高'].std()
##### (3)比例的点估计
#f=BSdata['开设'].value_counts();p=f/sum(f);p
42/150

#3.2.1.2 区间估计
da.norm_p(-2,2)

def t_interval(b,x):
    a=1-b
    n = len(x)
    import scipy.stats as st
    ta=st.t.ppf(1-a/2,n-1);ta
    from math import sqrt
    se=x.std()/sqrt(n)
    return(x.mean()-ta*se, x.mean()+se*ta)
t_interval(0.95,BSdata['身高'])

X=BSdata['身高']
st.norm.interval(0.95,X.mean())
n=len(X)
st.t.interval(0.95, n-1, X.mean(), X.std()/sqrt(n))
#st.t.interval(0.95, len(X)-1, X.mean(), st.sem(X))

###6.3.2 参数的假设检验
#####（1）t检验
import scipy.stats as st  #加载统计方法包
st.ttest_1samp(BSdata.身高, popmean = 166)
st.ttest_1samp(BSdata.身高, popmean = 170)
### 单样本t检验及图示
import DaPy_fun as da
da.ttest_1plot(BSdata.身高,166)
da.ttest_1plot(BSdata.身高,170)

#7 数据的模型分析
##7.1 简单线性相关模型
###7.1.1 线性相关的概念
x=np.linspace(-4,4,20); e=np.random.randn(20) #随机误差
fig,ax=plt.subplots(2,2,figsize=(14,12))
ax[0,0].plot(x,x,'o')
ax[0,1].plot(x,-x,'o')
ax[1,0].plot(x,x+e,'o');
ax[1,1].plot(x,-x+e,'o');

###7.1.2 相关系数的计算
#####（1）散点图
x=BSdata.身高;y=BSdata.体重
plt.plot(x, y,'o'); #plt.scatter(x,y);

#####（2）相关系数
x.cov(y)

x.corr(y)
y.corr(x)

###7.1.3 相关系数的检验
#####(3) 计算值和值，作结论。
st.pearsonr(x,y)  #pearson相关及检验

##7.2 简单线性回归模型
###7.2.1 简单线性模型估计
#####（1）模拟直线回归模型
#da.reglinedemo()
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

reglinedemo();
reglinedemo(50)

import statsmodels.api as sm             #简单线性回归模型
fm1=sm.OLS(y,sm.add_constant(x)).fit()   #普通最小二乘，家常数项
fm1.params                               #系数估计
yfit=fm1.fittedvalues;
plt.plot(x, y,'.',x,yfit, 'r-');

###7.2.2 简单线性模型检验
#####
fm1.tvalues                            #系数t检验值
fm1.pvalues                            #系数t检验概率
pd.DataFrame({'b估计值':fm1.params,'t值':fm1.tvalues,'概率p':fm1.pvalues})

import statsmodels.formula.api as smf  #根据公式建回归模型
fm2=smf.ols('体重~身高', BSdata).fit()
pd.DataFrame({'b估计值':fm2.params,'t值':fm2.tvalues,'概率p':fm2.pvalues})
fm2.summary2().tables[1]           #回归系数检验表
plt.plot(BSdata.身高,BSdata.体重,'.',BSdata.身高,fm3.fittedvalues,'r-');

###7.2.3 简单线性模型预测
fm2.predict(pd.DataFrame({'身高': [178,188,190]}))   #预测

##7.3 分组线性相关与回归
#####（1）绘制分组散点图
BS_M=BSdata[BSdata.性别=='男'][['身高','体重']];BS_M
BS_F=BSdata[BSdata.性别=='女'][['身高','体重']];BS_F

#####（2）分组相关分析
#plt.plot(BS_M.身高,BS_M.体重,'o');
import scipy.stats as st
st.pearsonr(BS_M.身高,BS_M.体重)
import seaborn as sns
sns.jointplot('身高','体重',BS_M)

#plt.plot(BS_F.身高, BS_F.体重,'o')
st.pearsonr(BS_F.身高,BS_F.体重)
sns.jointplot('身高','体重',BS_F)

smf.ols('体重~身高',BS_M).fit().summary2().tables[1]
sns.jointplot('身高','体重',BS_M,kind='reg')

smf.ols('体重~身高',BS_F).fit().summary2().tables[1]
sns.jointplot('身高','体重',BS_F,kind='reg')


#8 数据的预测分析
#8.1 动态数列的基本分析
QTdata=pd.read_excel('DaPy_data.xlsx','QTdata',index_col=0);QTdata.head(8)
QTdata.plot()
QTdata['Year']=QTdata.index.str[:4];QTdata
YGDP=QTdata.groupby(['Year'])['GDP'].sum();YGDP
YGDP.plot();

##8.1.2 动态数列的分析
YGDPds=pd.DataFrame(YGDP);YGDPds  #构建年度动态序列框
YGDPds['定基数']=YGDP-YGDP[:1].values;YGDPds
YGDPds['环基数']=YGDP-YGDP.shift(1);YGDPds  #shift(1)向下移动1个单位
YGDPds['定基比']=YGDP/YGDP[:1].values;YGDPds
YGDPds['环基比']=(YGDP/YGDP.shift(1)-1)*100;YGDPds

#Qt.index=pd.period_range('2001Q1','2015Q4',freq='Q');Qt  #形成季度数据
#Qt.plot();
QGDP=QTdata.GDP
QGDPds=pd.DataFrame({'GDP':QGDP});QGDPds
QGDPds['同比数']=QGDP-QGDP.shift(4); QGDPds

QGDPds['同基比']=(QGDP/QGDP.shift(4)-1)*100;QGDPds

n=1/len(YGDP)
ADR=(YGDP[-1:].values/YGDP[:1].values)**n
print('\n\t平均增长量 = %5.3f' % ADR)

#将上述过程构成一个动态数列函数并用于季度数据
#dyns(Qt)[:12]

#8.2 动态数列预测分析
x=np.arange(20)+1;x
y=1+0.2*x
plt.plot(x,1+0.2*x,'.');
plt.plot(x,1+0.2*np.log(x),'.');
plt.plot(x,0.2*np.exp(0.1*x),'.');
plt.plot(x,0.2*x**0.1,'.');

# 年度数据趋势分析
Yt=QTdata.groupby(['Year'])['GDP'].sum();Yt  #形成年度时序数据
plt.plot(Yt,'o')  #Yt.plot();

#1.线性趋势模型
import statsmodels.api as sm
Yt=YGDP                        #Yt=QTdata.groupby(['Year'])['GDP'].sum()
X1=np.arange(len(Yt))+1;X1     #自变量序列,建模时最好不直接用年份
Yt_L1=sm.OLS(Yt,sm.add_constant(X1)).fit();
Yt_L1.summary2().tables[1]
#plt.plot(X1,Yt,'o',X1,Yt_L1.fittedvalues,'r-');
#plt.plot(Yt,'o',Yt_L1.fittedvalues,'r-');

import warnings   #忽视警告信息
warnings.filterwarnings("ignore")
def trendmodel(y,x):  #定义两变量直线趋势回归模型，x自变量，y因变量
    fm=sm.OLS(y,sm.add_constant(x)).fit()
    sfm=fm.summary2()
    print("模型检验:\n",sfm.tables[1])
    print("决定系数：",sfm.tables[0][1][6])
    return fm.fittedvalues

L1=trendmodel(Yt,X1);
plt.plot(Yt,'o',L1,'r-');

#2.非线性趋势模型
#(3)指数模型
L2=trendmodel(np.log(Yt),X1);
plt.plot(Yt,'o',np.exp(L2),'r-');

#plt.plot(Yt,'o',L1,'y-',L2,'r-',L3,'b-',np.exp(L2),'g-');

##8.2.2 平滑预测方法
Qt=QTdata.GDP;Qt
Qt.mean()  #季节数据的平均

QtM=pd.DataFrame(Qt);QtM
QtM['M2']=Qt.rolling(3).mean();QtM  #2阶移动平均
QtM.plot()
QtM['M4']=Qt.rolling(5).mean();QtM  #4阶移动平均
QtM.plot()

#8.2.2.2 指数平滑预测法
QtE=pd.DataFrame(Qt);QtE
QtE['E3']=Qt.ewm(alpha=0.3).mean(); QtE   #平滑系数=0.3
#QtE.plot()
QtE['E8']=Qt.ewm(alpha=0.8).mean(); QtE   #平滑系数=0.8
QtE.plot();

##8.3 股票数据统计分析
stock=pd.read_excel('DaPy_data.xlsx','Stock',index_col=0);
stock.info()

stock.columns

stock=stock.dropna() # 由于数据中有15 个缺失值，需删除缺失数据NA
stock.info()

round(stock.describe(),3)

###8.3.1 股票价格分析

stock[['Close','Volume']].head()  #收盘价与成交量数据
stock['2015']['Close'].head()     #年度收盘价数据
stock['2015-10']['Close']         #月度收盘价数据

stock['Close'].plot();
stock['2015']['Close'].plot();
stock['Volume'].hist()

stock[['Close','Volume']].plot(secondary_y='Volume')
stock['2015'][['Close','Volume']].plot(secondary_y='Volume')

SC=stock['2015']['Close']; SC  #2015年收盘价数据
###移动平均线：
SCM=pd.DataFrame(SC);SCM
SCM['MA5']=SC.rolling(5).mean();
SCM['MA20']=SC.rolling(20).mean();
SCM['MA60']=SC.rolling(60).mean();
SCM
SCM.plot();
SCM.plot(subplots=False,figsize=(15,10),grid=True);
SCM.plot(subplots=True,figsize=(15,20),grid=True);

###8.3.2 股票收益率分析
def Return(Yt):   #计算收益率
    Rt=Yt/Yt.shift(1)-1  #Yt.diff()/Yt.shift(1)
    return(Rt)

SA=stock['2015']['Adjusted']; SA[:10]  #2015年调整价数据
SA_R=Return(SA);SA_R[:10]
SA_R.plot().axhline(y=0)

YR=pd.DataFrame({'Year':stock.index.year,'Adjusted':Return(stock['Adjusted'])});YR[:10]
YRm=YR.groupby(['Year']).mean();YRm
YRm.plot(kind='bar').axhline(y=0)

YMR=pd.DataFrame({'Year':stock.index.year,'Month':stock.index.month,
                  'Adjusted':Return(stock['Adjusted'])}); YMR[:10]
YMRm=YMR.groupby(['Year','Month']).mean(); YMRm[:15]
round(YMRm.unstack(),4)
YMRm.plot().axhline(y=0)

MRm=YMR['2005'].groupby(['Month']).mean(); MRm['Adjusted'].plot(kind='bar').axhline(y=0)

# 9 数据的决策分析
# 9.1 确定性决策分析 ----
## 9.1.1 单目标求解
Tv=pd.read_excel('DaPy_data.xlsx','Target',index_col=0); Tv  #目标值
Tv['年收益']=Tv.年销售量*(Tv.销售单价-Tv.单件成本)-Tv.设备投资;Tv
Tv['年收益'].idxmax()  # 最佳方案

## 9.1.2 多目标求解 ----
Ev=[min(Tv.设备投资),min(Tv.单件成本),max(Tv.年销售量),max(Tv.销售单价),
    max(Tv.年收益)];Ev #理想值
Dv=((Tv-Ev)**2).sum(1); #差值
pd.concat([Tv,Dv],axis=1)
Dv.idxmin()

# 9.2 不确定性决策分析 ----
## 9.2.1 分析方法
PLm=pd.DataFrame();PLm  # 损益矩阵 ProfitLoss matrix
PLm['畅销']= 12000*(Tv.销售单价-Tv.单件成本)-Tv.设备投资;
PLm['一般']= 8000*(Tv.销售单价-Tv.单件成本)-Tv.设备投资;
PLm['滞销']= 1500*(Tv.销售单价-Tv.单件成本)-Tv.设备投资;PLm

## 9.2.2 分析原则----
# 乐观原则
lg=PLm.max(1); pd.concat([PLm,lg],axis=1)
lg.idxmax()
# 悲观原则
bg=PLm.min(1); pd.concat([PLm,bg],axis=1)
bg.idxmax()
# 折中原则
a=0.65
zz= a*lg + (1-a)*bg; pd.concat([PLm,zz],axis=1)
zz.idxmax()
# 后悔原则
Rm=PLm.max()-PLm;Rm  #后悔矩阵 Regret matrix
hh=Rm.max(1); pd.concat([Rm,hh],axis=1)
hh.idxmin()

# 8.5 概率性决策分析 ----
## 8.5.1 期望值法 ----
PLm  # 损益矩阵
probE=[0.1,0.65,0.25]; #初始概率
qw=(probE*PLm).sum(1); pd.concat([PLm,qw],axis=1)
qw.idxmax()

# 8.5.2 后悔期望值法
Rm  # 后悔矩阵
probE=[0.1,0.65,0.25];
hhqw=(probE*Rm).sum(1); pd.concat([Rm,hhqw],axis=1)
hhqw.idxmin()

#10 数据的案例分析
##10.1 网上数据获取与保存
### 10.1.1 网上数据的获取
import tushare as ts   #python财经数据接口包 http://tushare.org

#沪深上市公司基本情况分析
?ts.get_stock_basics()
s_b=ts.get_stock_basics();
s_b.info()
s_b.head()
### 10.1.2 在线股票数据分析
s_b.area.value_counts()
s_b.area.value_counts().plot(kind='barh')
s_b.industry.value_counts()
s_b.industry.value_counts()[:20].plot(kind='barh'); #前20个行业分布
s_b.groupby(['industry'])[['pe','pb','esp','gpr','npr']].mean()[:10]
esp_ind=s_b.groupby(['industry'])['esp'].mean();esp_ind #按行业(industry)计算平均收益率(esp)
esp_ind.sort_values().head(10) #收益率最差的10个行业
esp_ind.sort_values().tail(10) #收益率最好的10个行业
esp_ind.sort_values().head(10).plot(kind='bar')
esp_ind.sort_values().tail(10).plot(kind='bar')
#按地区(area)和行业(industry)计算平均收益率(esp)  #,gpr(毛利率(%)),npr(净利润率(%)),pe(市盈率)
esp_ind_area=s_b.groupby(['area','industry'])['esp'].mean(); esp_ind_area
esp_ind_area['广东'].sort_values().head(10) #广东省收益率最差的10个行业
esp_ind_area['广东'].sort_values().tail(10) #广东省收益率最好的10个行业
#s_b.pivot_table('esp','industry','area',aggfunc=[np.mean])

### 10.1.3 新股发行数据分析
ts.new_stocks()
n_s=ts.new_stocks();n_s
n_s.info()
n_s18=n_s.loc[n_s.ipo_date>='2018',];n_s18  #2018年10月30日前发行的新股
n_s18.sort_values(by='amount').iloc[-10:,:6]  #18年10月30日前发行量最大的10只新股
n_s18.sort_values(by='ballot').iloc[-10:,[0,1,2,3,4,5,11]] #18年6月1日前中签率最高的10只新股
plt.plot(n_s18.amount,n_s18.ballot,'o'); #发行量和中签率之间的散点图
n_s18.amount.corr(n_s18.ballot)          #发行量和中签率之间的相关系数




##10.2 证券交易数据的分析
### 10.2.1 股票行情数据分析
#### 10.2.1 历史行情数据分析
#hs300
h_s=ts.get_hist_data('399300')  #'hs300' 沪深300指数近三年的历史行情数据
#ts.get_hist_data('399300',start='2018-01-01',end='2018-12-31') #指定时间区间
h_s.info()
h_s.columns
h_s.head()
h_s.sort_index(inplace=True); #按时间排序
h_s.head()
h_s['close'].to_csv('hs300.csv')

h_s['close'].plot()
h_s['volume'].plot()
h_s['price_change'].plot().axhline(y=0,color='red')
h_s['p_change'].plot().axhline(y=0,color='red')

h_s[['open','close']].plot()
h_s[['open','close','high','low']].plot()
h_s[['close','ma5','ma10','ma20']].plot()

#### 10.2.2 实时行情数据分析
t_a=ts.get_today_all()
t_a.info()
t_a.head()
down=t_a['changepercent'].sort_values().head(10).index #跌幅最大的10个行业
t_a.loc[down,['code','name','changepercent','trade','settlement','turnoverratio']]
up=t_a['changepercent'].sort_values().t_ail(10).index   #涨幅最大的10个行业
t_a.loc[up,['code','name','changepercent','trade','settlement','turnoverratio']]


#### 10.2.3 大单交易数据分析
s_d=ts.get_sina_dd('002024', date='2019-10-30',vol=400) #默认400手
s_d
s_d.info()
s_d.head(10)
da.tab(s_d['type'])
s_d['type'].value_counts().plot(kind='pie');

#### 10.2.4 公司盈利能力分析
p_d=ts.get_profit_data(2018,1);p_d   #获取2018年第1季度的盈利能力数据
p_d.info()
p_d.columns=['代码','名称','净收益率','净利润率','毛利润率','净利润额','每股收益','营业收入','主营收入']
round(p_d.head(10),3)
round(p_d.describe(),2) #基本统计分析
round(p_d.corr(),3)     #相关性分析

####10.2.5 公司现金流量分析
c_a=ts.get_c_ashflow_data(2018,1)  #获取2018年第1季度的现金流量数据
c_a.info()
c_a.head()
st=c_a['name'].str[:3]=='*ST'  #选取ST公司
c_a.loc[st,].sort_values(by='c_ashflowratio').head(10)  #现金流量比率最差的10家ST公司
c_a.loc[st,].sort_values(by='cashflowratio').tail(10)  #现金流量比率最好的10家ST公司

##10.3 宏观经济数据的实证分析
###10.3.1 存款利率变动分析
d_r=ts.get_deposit_rate()
d_r.info()
d_r
d_r.deposit_type.value_counts()

dr1=d_r[d_r.deposit_type=='活期存款(不定期)'].sort_values(by='date');
dr1.index=dr1.date.str[:7];dr1
dr2=d_r[d_r.deposit_type=='定期存款整存整取(一年)'].sort_values(by='date');
dr2.index=dr2.date.str[:7];dr2
dr3=pd.concat([dr1.rate.astype(float),dr2.rate.astype(float)],axis=1);
dr3.columns=['活期存款(不定期)','整存整取(一年)'];dr3
dr3.plot();
dr3.plot(secondary_y='整存整取(一年)');

###10.3.2 国内生产总值GDP分析
g_y=ts.get_gdp_year()  ## 国内生产总值(年度)
g_y.info()
g_y.head()
#g_y.index=g_y.year
#g_y.drop(['year'],axis=1,inplace=True)
#g_y.sort_index(inplace=True) #
g_y.sort_values(by='year',inplace=True)
g_y.head()
plt.plot(g_y.year,g_y.gdp)

g_y1=g_y[g_y.year>=1990];g_y1
plt.plot(g_y1.year,g_y1.gdp)

g_y2=g_y1[['pi','si','ti']]

g_y2.index=g_y1.year; g_y2
g_y2.plot(kind='bar')
g_y2.plot(kind='line')

###10.3.3 工业品出厂价格指数分析
g_p=ts.get_ppi()
g_p.info()
g_p
g_p.sort_values(by='month',inplace=True); g_p
g_p.index=g_p.month;g_p
g_p.plot();
#工业品价格指数
g_p1=g_p[['ppiip','ppi','qm','rmi','pi']].dropna()
g_p1.plot();
#生活价格指数
g_p2=g_p[['cg','food','clothing','roeu','dcg']].dropna();g_p2
g_p2.plot(grid=True)

###10.4 电影票房数据的实时分析
#实时票房
#获取实时电影票房数据，30分钟更新一次票房数据，可随时调用。
r_b = ts.realtime_boxoffice()
r_b.info()
r_b

plt.barh(r_b.MovieName,r_b.BoxOffice.astype(float));
plt.pie(r_b.boxPer,labels=r_b.MovieName);

#每日票房
d_b = ts.day_boxoffice() #取上一日的数据
d_b

#影院日度票房
#获取全国影院单日票房排行数据，默认为上一日，可输入日期参数获取指定日期的数据。
d_c=ts.day_cinema() #取上一日全国影院票房排行数据
d_c.info()

d_c[:10]
plt.barh(d_c.CinemaName[:10],d_c.Attendance.astype(float)[:10]);
