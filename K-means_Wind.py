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
from sklearn.cluster import KMeans
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
qWeather='SELECT StnPres,SeaPres,Temperature,DewPointTemperature,RelativeHumidity,WindSpeed,WindDirection,WSGust,WDGust,Precipitation,PrecpHour FROM codis where year(DataTime) '
#queryTime='SELECT DataTime FROM codis0202 where year(DataTime) = 2015 and 2016'
qPower='SELECT Grid_Power FROM wind_turbines where year(DATE_TIME) and CK_NUMBER= 3;'

df_Weather=pd.read_sql(qWeather,con=db)
df_Power=pd.read_sql(qPower,con=db)

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.3)   # parse train 70% , test 30% ,random_state 固定隨機數種子

X=df_Weather
y=df_Power


imp_1=preprocessing.Imputer(missing_values='NaN',strategy='mean',axis=0)      #預處理，空值改均值 axis=0
imp_1.fit(X)
X=imp_1.transform(X)

imp_2=preprocessing.Imputer(missing_values='NaN',strategy='mean',axis=0)      #預處理，空值改均值 axis=0
imp_2.fit(y)
y=imp_2.transform(y)

#y=y.reshape(-1)


#特徵縮放
#-----------------------------------------------------------------------------------------------------------------------
sc=preprocessing.StandardScaler()
sc.fit(X)
X=sc.fit_transform(X)
#-----------------------------------------------------------------------------------------------------------------------

wcss=[]
for i in range(1,51):
    Kmeans = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=None,
                    algorithm='auto')

    X_pred = Kmeans.fit(X)
    wcss.append(X_pred.inertia_)


plt.figure(figsize=(12,8))
plt.plot(range(1,51),wcss)
plt.title('Elbow Method')
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.show()
