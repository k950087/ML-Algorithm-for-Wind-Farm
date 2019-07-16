from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.ensemble import RandomForestRegressor
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

# and month(DataTime) not between 4 and 11 and WindDirection not between 100 and 350  and WindDirection!=0
# and hour(DataTime) between 10 and 15   ,ROTO_ROTORRPM

qWeather = 'SELECT StnPres,SeaPres,Temperature,DewPointTemperature,RelativeHumidity,WindSpeed,WindDirection,WSGust,WDGust,ROTO_ROTORRPM FROM ck5_combine where hour(DataTime) between 10 and 15'
qPower = 'SELECT Grid_Power FROM ck5_combine where hour(DataTime) between 10 and 15'


df_Weather=pd.read_sql(qWeather,con=db)
df_Power=pd.read_sql(qPower,con=db)


X=df_Weather
y=df_Power


imp_1=preprocessing.Imputer(missing_values='NaN',strategy='mean',axis=0)      #預處理，空值改均值 axis=0
imp_1.fit(X)
X=imp_1.transform(X)

imp_2=preprocessing.Imputer(missing_values='NaN',strategy='mean',axis=0)      #預處理，空值改均值 axis=0
imp_2.fit(y)
y=imp_2.transform(y)

y=y.reshape(-1)


#特徵縮放
#-----------------------------------------------------------------------------------------------------------------------
sc=preprocessing.StandardScaler()
sc.fit(X)
X=sc.transform(X)
#-----------------------------------------------------------------------------------------------------------------------


#rbf_svr = SVR(kernel='rbf', C=1e3)
RF=RandomForestRegressor(n_estimators=10,
                         criterion='mse',
                         random_state=14
                         )

sfs = SFS(RF,
          k_features=10,
          forward=True,
          floating=False,
          scoring='r2',                 #{'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error', 'median_absolute_error', 'r2'}
          cv=10) # n_jobs=-1 means all CPUs

sfs = sfs.fit(X, y)
fig = plot_sfs(sfs.get_metric_dict(), kind='std_dev')   #{'std_dev', 'std_err', 'ci', None}.

plt.title('Sequential Forward Selection (季風)')
plt.grid()
plt.show()
