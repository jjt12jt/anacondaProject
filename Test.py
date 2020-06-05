from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from pandas import read_csv
import pandas as pd

sourceData=read_csv('/Users/jiangjiantao/Downloads/79079699_2_py大作业_调查_5_5.csv',encoding='GBK')
print(sourceData)
y=read_csv('/Users/jiangjiantao/Downloads/yyyyyyyyyyyyyyy.csv',encoding='GBK')
print(y.shape)

XTrain, XTest, YTrain, YTest = train_test_split(sourceData,y,test_size=0.6,random_state=420)
for i in [XTrain, XTest]:
    i.index = range(i.shape[0])
XTrain.shape

reg=LR().fit(XTrain,YTrain)
yHat=reg.predict(XTest)
print(yHat)


reg.coef_
[*zip(XTrain.columns,reg.coef_)]
print(reg.intercept_)

from sklearn.metrics import mean_squared_error as MSE
print(MSE(yHat,YTest))
y.max()
y.min()
import sklearn
sorted(sklearn.metrics.SCORERS.keys())
print(cross_val_score(reg,sourceData,y,cv=5,scoring="neg_mean_squared_error"))