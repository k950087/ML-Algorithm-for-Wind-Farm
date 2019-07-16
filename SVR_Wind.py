from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
#from sklearn.utils import check_arrays
#-----------------------------------------------------------------------------------------------------------------------
acc='abc'
pwd=1234
dbName='data_analysis'                 #資料庫
tbName_1='codis'
tbName_2='wind_turbines'                    #資料表
ip='140.120.54.129'
port=3306
folderPath='C:\CK_WIND\CK_4'         #文件夾路徑
#-----------------------------------------------------------------------------------------------------------------------
db = create_engine('mysql://'+acc+':'+str(pwd)+'@'+ip+':'+str(port)+'/'+dbName) #connect mysql

#CK3 accuracy 最高
qTrain_Weather='SELECT DewPointTemperature,RelativeHumidity,WindSpeed,WindDirection,WSGust FROM codis where year(DataTime) <= 2016'
qTest_Weather='SELECT DewPointTemperature,RelativeHumidity,WindSpeed,WindDirection,WSGust FROM codis where year(DataTime) = 2017'
#queryTime='SELECT DataTime FROM codis0202 where year(DataTime) = 2015 and 2016'
qTrain_Power='SELECT Grid_Power FROM wind_turbines where year(DATE_TIME) <= 2016 and CK_NUMBER= 3;'
qTest_Power='SELECT Grid_Power FROM wind_turbines where year(DATE_TIME) = 2017 and CK_NUMBER= 3;'

df_Train_Weather=pd.read_sql(qTrain_Weather,con=db)
df_Test_Weather=pd.read_sql(qTest_Weather,con=db)
#DataTime=pd.read_sql(queryTime,con=db_1)
df_Train_Power=pd.read_sql(qTrain_Power,con=db)
df_Test_Power=pd.read_sql(qTest_Power,con=db)

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.3)   # parse train 70% , test 30% ,random_state 固定隨機數種子

X_train=df_Train_Weather
X_test=df_Test_Weather
y_train=df_Train_Power
y_test=df_Test_Power


imp_1=preprocessing.Imputer(missing_values='NaN',strategy='mean',axis=0)      #預處理，空值改均值 axis=0
imp_1.fit(X_train)
X_train=imp_1.transform(X_train)
X_test=imp_1.transform(X_test)

imp_2=preprocessing.Imputer(missing_values='NaN',strategy='mean',axis=0)      #預處理，空值改均值 axis=0
imp_2.fit(y_train)
y_train=imp_2.transform(y_train)
y_test=imp_2.transform(y_test)

y_train=y_train.reshape(-1)
y_test=y_test.reshape(-1)

#特徵縮放
#-----------------------------------------------------------------------------------------------------------------------
sc=preprocessing.StandardScaler()
sc.fit(X_train)
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
#-----------------------------------------------------------------------------------------------------------------------


#linear=LinearRegression()
rbf_svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
#lin_svr = SVR(kernel='linear', C=1e3)
#poly_svr = SVR(kernel='poly', C=1e3)


#model_linear=linear.fit(X_train, y_train)
model_svr_rbf=rbf_svr.fit(X_train, y_train)
#model_svr_lin=lin_svr.fit(X_train, y_train)
#model_svr_poly=poly_svr.fit(X_train, y_train)

#y_pred_linear=model_linear.predict(X_test)
y_pred_svr_rbf=model_svr_rbf.predict(X_test)
#y_pred_svr_lin=model_svr_lin.predict(X_test)
#y_pred_svr_poly=model_svr_poly.predict(X_test)

#MSE=metrics.mean_squared_error(y_test, y_pred_linear)
MSE_rbf=metrics.mean_squared_error(y_test, y_pred_svr_rbf)
#MSE_lin=metrics.mean_squared_error(y_test, y_pred_svr_lin)
#MSE_poly=metrics.mean_squared_error(y_test, y_pred_svr_poly)

#RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_pred_linear))
RMSE_rbf=np.sqrt(metrics.mean_squared_error(y_test, y_pred_svr_rbf))
#RMSE_lin=np.sqrt(metrics.mean_squared_error(y_test, y_pred_svr_lin))
#RMSE_poly=np.sqrt(metrics.mean_squared_error(y_test, y_pred_svr_poly))


#ACC=linear.score(X_test,y_test)
ACC_rbf=rbf_svr.score(X_test,y_test)
#ACC_lin=lin_svr.score(X_test,y_test)
#ACC_poly=poly_svr.score(X_test,y_test)

# MAPE
#-----------------------------------------------------------------------------------------------------------------------
#y_test=np.where(y_test == 0 , 0.00000000001 , y_test)
def mean_absolute_percentage_error(a, b):
    mask = a != 0
    return (abs(a-b)/a)[mask].mean()
'''
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))
'''
#-----------------------------------------------------------------------------------------------------------------------
#MAPE_rbf=mean_absolute_percentage_error(y_test,y_pred_svr_rbf)

print('*'*100)
print('MSE :',MSE_rbf)
print('RMSE :',RMSE_rbf)
#print('MAPE :',MAPE_rbf)
print('Accuracy :' , ACC_rbf)
#print('lin kernel :' , ACC_lin)
#print('poly kernel :' , ACC_poly)
print('*'*100)

#cross_val = cross_val_predict(model_linear, X, y, cv=10)
#cross_val_2 = cross_val_predict(model_rbf_lin, X, y, cv=10)


#plt.scatter(DataTime, y, color='darkorange', label='data')
#plt.plot(DataTime, y, color='darkorange', label='true')
#plt.plot(DataTime, model_svr_rbf.predict(X), color='black',linestyle='--', label='predict')
#plt.plot(y_pred_svr_rbf, y_test)
#plt.plot(DataTime, y_pred_svr_rbf)
#plt.scatter(y, cross_val)
#plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', lw=4)
#plt.xlabel('Time')
#plt.ylabel('Power generation')
#plt.legend(loc='upper right')
#plt.show()
