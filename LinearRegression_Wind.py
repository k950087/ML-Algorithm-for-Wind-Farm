from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
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


#qTrain_Weather='SELECT DewPointTemperature,RelativeHumidity,WindSpeed,WindDirection,WSGust FROM codis where year(DataTime) <= 2016 '    #SFS
#qTrain_Weather='SELECT Temperature,DewPointTemperature,RelativeHumidity,WindSpeed,WSGust FROM codis where year(DataTime) <= 2016 '         #RandomForest
#qTrain_Weather='SELECT StnPres,SeaPres,Temperature,DewPointTemperature,RelativeHumidity,WindSpeed,WindDirection,WSGust,WDGust,Precipitation,PrecpHour FROM codis where year(DataTime) <= 2016 '
qTrain_Weather='SELECT StnPres,SeaPres,Temperature,DewPointTemperature,RelativeHumidity,WindSpeed,WindDirection,WSGust,WDGust FROM codis where year(DataTime) <= 2016 '

#qTrain_Weather='SELECT StnPres,SeaPres,Temperature,DewPointTemperature,RelativeHumidity,WindSpeed,WindDirection,WSGust,WDGust FROM codis where year(DataTime) <= 2016 and month(DataTime) not between 5 and 9 and hour(DataTime) between 11 and 15'
#qTrain_Weather='SELECT StnPres,SeaPres,Temperature,DewPointTemperature,RelativeHumidity,WindSpeed,WindDirection,WSGust,WDGust FROM codis where year(DataTime) <= 2016 and month(DataTime) not between 5 and 9'
#qTrain_Weather='SELECT StnPres,SeaPres,Temperature,DewPointTemperature,RelativeHumidity,WindSpeed,WindDirection,WSGust,WDGust,Precipitation,PrecpHour FROM codis where year(DataTime) <= 2016 and hour(DataTime) between 11 and 15'
#qTrain_Weather='SELECT StnPres,SeaPres,Temperature,DewPointTemperature,RelativeHumidity,WindSpeed,WindDirection,WSGust,WDGust,Precipitation,PrecpHour FROM codis where year(DataTime) <= 2016 and hour(DataTime) not between 5 and 22'



qTrain_ROTO_ROTORRPM='SELECT ROTO_ROTORRPM FROM wind_turbines where year(DATE_TIME) <= 2016 and CK_NUMBER= 31'
#qTrain_ROTO_ROTORRPM='SELECT ROTO_ROTORRPM FROM wind_turbines where year(DATE_TIME) <= 2016 and CK_NUMBER= 3 and month(DATE_TIME) not between 5 and 9 and hour(DATE_TIME) between 11 and 15'
#qTrain_ROTO_ROTORRPM='SELECT ROTO_ROTORRPM FROM wind_turbines where year(DATE_TIME) <= 2016 and CK_NUMBER= 3 and month(DATE_TIME) not between 5 and 9'
#qTrain_ROTO_ROTORRPM='SELECT ROTO_ROTORRPM FROM wind_turbines where year(DATE_TIME) <= 2016 and CK_NUMBER= 3 and month(DATE_TIME) and hour(DATE_TIME) between 11 and 15'
#qTrain_ROTO_ROTORRPM='SELECT ROTO_ROTORRPM FROM wind_turbines where year(DATE_TIME) <= 2016 and CK_NUMBER= 3 and month(DATE_TIME) and hour(DATE_TIME) not between 5 and 22'


#qTest_Weather='SELECT DewPointTemperature,RelativeHumidity,WindSpeed,WindDirection,WSGust FROM codis where year(DataTime) = 2017' #SFS
#qTest_Weather='SELECT Temperature,DewPointTemperature,RelativeHumidity,WindSpeed,WSGust FROM codis where year(DataTime) = 2017' #RandomForest
#qTest_Weather='SELECT StnPres,SeaPres,Temperature,DewPointTemperature,RelativeHumidity,WindSpeed,WindDirection,WSGust,WDGust,Precipitation,PrecpHour FROM codis where year(DataTime) = 2017'
qTest_Weather='SELECT StnPres,SeaPres,Temperature,DewPointTemperature,RelativeHumidity,WindSpeed,WindDirection,WSGust,WDGust FROM codis where year(DataTime) = 2017'

#qTest_Weather='SELECT StnPres,SeaPres,Temperature,DewPointTemperature,RelativeHumidity,WindSpeed,WindDirection,WSGust,WDGust FROM codis where year(DataTime) = 2017 and month(DataTime) not between 5 and 9 and hour(DataTime) between 11 and 15'
#qTest_Weather='SELECT StnPres,SeaPres,Temperature,DewPointTemperature,RelativeHumidity,WindSpeed,WindDirection,WSGust,WDGust FROM codis where year(DataTime) = 2017 and month(DataTime) not between 5 and 9'
#qTest_Weather='SELECT StnPres,SeaPres,Temperature,DewPointTemperature,RelativeHumidity,WindSpeed,WindDirection,WSGust,WDGust,Precipitation,PrecpHour FROM codis where year(DataTime) = 2017 and hour(DataTime) between 11 and 15'
#qTest_Weather='SELECT StnPres,SeaPres,Temperature,DewPointTemperature,RelativeHumidity,WindSpeed,WindDirection,WSGust,WDGust,Precipitation,PrecpHour FROM codis where year(DataTime) = 2017 and hour(DataTime) not between 5 and 22'


qTest_ROTO_ROTORRPM='SELECT ROTO_ROTORRPM FROM wind_turbines where year(DATE_TIME) = 2017 and CK_NUMBER= 31'
#qTest_ROTO_ROTORRPM='SELECT ROTO_ROTORRPM FROM wind_turbines where year(DATE_TIME) = 2017 and CK_NUMBER= 3 and month(DATE_TIME) not between 5 and 9 and hour(DATE_TIME) between 11 and 15'
#qTest_ROTO_ROTORRPM='SELECT ROTO_ROTORRPM FROM wind_turbines where year(DATE_TIME) = 2017 and CK_NUMBER= 3 and month(DATE_TIME) not between 5 and 9'
#qTest_ROTO_ROTORRPM='SELECT ROTO_ROTORRPM FROM wind_turbines where year(DATE_TIME) = 2017 and CK_NUMBER= 3 and hour(DATE_TIME) between 11 and 15'
#qTest_ROTO_ROTORRPM='SELECT ROTO_ROTORRPM FROM wind_turbines where year(DATE_TIME) = 2017 and CK_NUMBER= 3 and hour(DATE_TIME) not between 5 and 22'


qTrain_Power='SELECT Grid_Power FROM wind_turbines where year(DATE_TIME) <= 2016 and CK_NUMBER= 31'
#qTrain_Power='SELECT Grid_Power FROM wind_turbines where year(DATE_TIME) <= 2016 and CK_NUMBER= 3 and month(DATE_TIME) not between 5 and 9 and hour(DATE_TIME) between 11 and 15'
#qTrain_Power='SELECT Grid_Power FROM wind_turbines where year(DATE_TIME) <= 2016 and CK_NUMBER= 3 and month(DATE_TIME) not between 5 and 9;'
#qTrain_Power='SELECT Grid_Power FROM wind_turbines where year(DATE_TIME) <= 2016 and CK_NUMBER= 3 and hour(DATE_TIME) between 11 and 15;'
#qTrain_Power='SELECT Grid_Power FROM wind_turbines where year(DATE_TIME) <= 2016 and CK_NUMBER= 3 and hour(DATE_TIME) not between 5 and 22'


qTest_Power='SELECT Grid_Power FROM wind_turbines where year(DATE_TIME) = 2017 and CK_NUMBER= 31'
#qTest_Power='SELECT Grid_Power FROM wind_turbines where year(DATE_TIME) = 2017 and CK_NUMBER= 3 and month(DATE_TIME) not between 5 and 9 and hour(DATE_TIME) between 11 and 15'
#qTest_Power='SELECT Grid_Power FROM wind_turbines where year(DATE_TIME) = 2017 and CK_NUMBER= 3 and month(DATE_TIME) not between 5 and 9;'
#qTest_Power='SELECT Grid_Power FROM wind_turbines where year(DATE_TIME) = 2017 and CK_NUMBER= 3 and hour(DATE_TIME) between 11 and 15;'
#qTest_Power='SELECT Grid_Power FROM wind_turbines where year(DATE_TIME) = 2017 and CK_NUMBER= 3 and hour(DATE_TIME) not between 5 and 22;'


df_Train_Weather=pd.read_sql(qTrain_Weather,con=db)
df_Train_ROTO_ROTORRPM=pd.read_sql(qTrain_ROTO_ROTORRPM,con=db)
df_Train_Combine=df_Train_Weather.join(df_Train_ROTO_ROTORRPM)


df_Test_Weather=pd.read_sql(qTest_Weather,con=db)
df_Test_ROTO_ROTORRPM=pd.read_sql(qTest_ROTO_ROTORRPM,con=db)
df_Test_Combine=df_Test_Weather.join(df_Test_ROTO_ROTORRPM)


#DataTime=pd.read_sql(queryTime,con=db_1)
df_Train_Power=pd.read_sql(qTrain_Power,con=db)
df_Test_Power=pd.read_sql(qTest_Power,con=db)

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.3)   # parse train 70% , test 30% ,random_state 固定隨機數種子

X_train=df_Train_Combine
X_test=df_Test_Combine
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
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)
#-----------------------------------------------------------------------------------------------------------------------

LR=LinearRegression()

model_LR=LR.fit(X_train, y_train)

y_pred=model_LR.predict(X_test)

MSE=metrics.mean_squared_error(y_test, y_pred)
RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
MAE=metrics.mean_absolute_error(y_test, y_pred)
R_Square=LR.score(X_test,y_test)


# MAPE
#-----------------------------------------------------------------------------------------------------------------------
def MAPE(y_true, y_pred):   #0值刪除
    actual=[]
    predict=[]
    for Check_Zero in range(y_true.shape[0]):
        if y_true[Check_Zero]!=0:
            actual.append(y_true[Check_Zero])
            predict.append(y_pred[Check_Zero])
    actual, predict = np.array(actual), np.array(predict)
    return np.mean(np.abs((actual - predict) / actual))

#-----------------------------------------------------------------------------------------------------------------------

print('*'*100)
print('天氣數據+風機轉子轉速')
print('MSE :',MSE)
print('RMSE :',RMSE)
print('MAPE :',MAPE(y_test,y_pred))
print('Accuracy :' , R_Square)
print('*'*100)
