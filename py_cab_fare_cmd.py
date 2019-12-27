#!/usr/bin/env python3
#Importing Libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from geopy.distance import great_circle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

def splitter(X,y):
       """fuction for test train spli"""
       X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                        test_size = 0.20,
                                                        random_state=20)
       return (X_train,X_test,y_train,y_test)

       


"""Importing dataset"""
os.chdir('D:\MY PROGRAMMING DATA\edwisor\Project02')
train = pd.read_csv("train_cab.csv")
test = pd.read_csv("test.csv")

#################################################################################

##########################----DATA Exploration----################################

###############################################################################

"""DATA Preprocessing"""
train.shape
#(16067, 7)
"""So our Train data has 16067 roes and 7 variables"""
test.shape
#(9914, 6)
"""The test data has shape of 9914 rows and 6 columns"""
"""So in this data set we have fare_amount as Dependent variable and rest 6 variables 
as Independent variable"""

###Univariate Analysis
train.dtypes
#fare_amount           object
#pickup_datetime       object
#pickup_longitude     float64
#pickup_latitude      float64
#dropoff_longitude    float64
#dropoff_latitude     float64
#passenger_count      float64
"""Here we can see that our fare_amount is object  and pickup_datetime is object 
type. Rest all the variables are in float. lets look at the each variable one by one.
So we first need to convert fare_amount in float and then convert passenger count in int 
as passenger count cannot be in float. also there are special characters in the fare_amount feature
, So we will replace them as well"""
#changing fare_amount in float and dropping negative values
train=train.drop(2486)
train=train.drop(2039)
train=train.drop(13032)
train.fare_amount = train.fare_amount.replace({"-":""},regex=True)
train.fare_amount = train.fare_amount.astype(float)
train = train.reset_index(drop=True)

#Changing pickup_datetime to date time format of train dataset
train.pickup_datetime[1327] = np.NAN
train.pickup_datetime = pd.to_datetime(train.pickup_datetime,format = '%Y-%m-%d %H:%M:%S UTC')
"""Thoiugh we have put a nan value as the value on that index was of simple int. So 
the conversion to datetime object of this variable is not fully and we will remove the nan value 
in removing NA section"""

"""Rest all variables are float,for longitude and latitudes its ok but not for passenger 
but as NA values are there in NA so we will cover this also under NA removal section"""

########################Univariate Analysis
test.dtypes
#pickup_datetime       object
#pickup_longitude     float64
#pickup_latitude      float64
#dropoff_longitude    float64
#dropoff_latitude     float64
#passenger_count        int64
"""Here the Passenger count is in int which is acceptable but pickup_datetime is in 
object so we need to convert it in date time object"""
#Changing pickup_datetime to date time format of test dataset
test.pickup_datetime = pd.to_datetime(test.pickup_datetime,format = '%Y-%m-%d %H:%M:%S UTC')

########################Dealing with Outliers and NA
train.isna().sum()
#fare_amount          24
#pickup_datetime       1
#pickup_longitude      0
#pickup_latitude       0
#dropoff_longitude     0
#dropoff_latitude      0
#passenger_count       0

test.isna().sum()
#pickup_datetime      0
#pickup_longitude     0
#pickup_latitude      0
#dropoff_longitude    0
#dropoff_latitude     0
#passenger_count      0

train.sort_values(by='fare_amount',ascending=False).head(20)
"""We have two values 54343 and 4343 in fare_amount but we cannot say that this can be an outlier.
Though 54343 can be an outlier but we cannot say with surerity"""
#Replacing 
x = train.fare_amount
x[1015] = np.nan
x.mean() # 11.6
train.fare_amount.mean() #15.04
"""We can see that replacing 54343 alue with NAN and then replacing then comparing the mean with 
unremoved 54343 value, there is not much differenc in the mean. So we will replace NAN values with mean"""
train.fare_amount.fillna(15.04,inplace = True)

#Replacing NA of pickup_datetime
train.pickup_datetime.fillna(method='ffill',inplace = True)

#boxplot Passenger_count
#lets see sort this varibale into descending order and then see what are those high values
train.sort_values(by='passenger_count',ascending=False).head(20)
"""So we go few values ranging from 35 to 5345 which is not possible for a cab service
So we need to replace these outliers"""
index_list = []
for i in range(len(train)):
       if train.passenger_count[i] > 6:
              index_list.append(i)
for i in range(len(train)):
       if train.passenger_count[i] <1:
              index_list.append(i)           
              
train = train.drop(index_list)
train = train.dropna()
train = train.reset_index(drop=True)
#rounding off the values in passenger count and removing NA
train.passenger_count = train.passenger_count.astype(int)

#Checking Na values if left
train.isna().sum()
#fare_amount          0
#pickup_datetime      0
#pickup_longitude     0
#pickup_latitude      0
#dropoff_longitude    0
#dropoff_latitude     0
#passenger_count      0

#Checking Longitude an latitude. Latitudes range from -90 to 90 and longitudes range from -180 to 80
#train
#Pickup
train.pickup_latitude.min() #-74.006893
train.pickup_latitude.max() # 401.083332
train.pickup_longitude.min() # -74.438233
train.pickup_longitude.max() # 40.766125
#Dropoff
train.dropoff_latitude.min() # -74.006377
train.dropoff_latitude.max() # 41.366138
train.dropoff_longitude.min() # --74.42933199999999
train.dropoff_longitude.max() # 40.802437
#train
#Pickup
test.pickup_latitude.min() #-40.573143
test.pickup_latitude.max() # 41.709555
test.pickup_longitude.min() # -74.252193
test.pickup_longitude.max() # -72.986532
#Dropoff
test.dropoff_latitude.min() # 40.568973
test.dropoff_latitude.max() # 41.696683
test.dropoff_longitude.min() # -74.263242
test.dropoff_longitude.max() # -72.990963

#deleting few cases where longitude nd latitude are irrelevant wrt to the data
train = train.drop(6927)# latitude 401
train = train.drop(1135) # lat approx to 0
train = train.drop(5782) #lon approx to -7
train = train.drop(5605)
train = train.drop(train[train['pickup_latitude']==0].index)
train = train.drop(train[train['dropoff_longitude']==0].index)
train = train.reset_index(drop=True)


"""For some of the lat lon. when investigated are pointing towards artic ocaen which is not relevant.
After investigating further, I got to to know that for those indexes the values of latitudes and longitudes 
have been interchnageed. If we interchnage them again then we will get locations of manhattan or Queens which
is much more relevant. So we will interchnage them"""
xyz = train[round(train['pickup_longitude']) == 41]
train.loc[xyz.index,['pickup_longitude','pickup_latitude']]=train.loc[xyz.index,['pickup_latitude','pickup_longitude']].values
train.loc[xyz.index,['dropoff_longitude','dropoff_latitude']]=train.loc[xyz.index,['dropoff_latitude','dropoff_longitude']].values
del xyz

"""Great we dont have any NA value now and our dataset is complete and now we can proceed """

#Adding Dataset
data = pd.concat([train[['pickup_datetime', 'pickup_longitude', 'pickup_latitude',
                         'dropoff_longitude', 'dropoff_latitude', 'passenger_count']],
       test])
data = data.reset_index(drop=True)

"""Replacing values with mee=an of that month"""
index_nan=[]
index_nan = list(train[train['fare_amount']>164].index)
for i in index_nan:
       train.fare_amount[i] = np.nan
       yr = train.pickup_datetime.dt.year[i]
       xyz=train[(train.pickup_datetime.dt.year)==yr]
       mn = xyz.pickup_datetime.dt.month[i]
       mean=xyz[(xyz.pickup_datetime.dt.month)==mn]['fare_amount'].mean()
       train.fare_amount[i] = mean
   
###############################################################################

##########################----Feature Enginnering---###########################
       
###############################################################################
#Creating Column of distance from pickup and dropoff
"""Using Great Circle Distance"""
x=[]
for i in range(len(data)):
       d1 = (data.pickup_latitude[i],data.pickup_longitude[i])
       d2 = (data.dropoff_latitude[i],data.dropoff_longitude[i])
       x.append(great_circle(d1, d2).km)
data['distance'] = x

#Dropping index where distance = 0
index_drop = list(data[data['distance'] ==0].index)

for index in index_drop:
       if index in train.index:
              train = train.drop(index)
              data = data.drop(index)

data = data.reset_index(drop=True)
train = train.reset_index(drop=True)
              
#Breaking Date Time and Year into different columns
data['year'] = data.pickup_datetime.dt.year
data['month']= data.pickup_datetime.dt.month 
data['day'] = data.pickup_datetime.dt.day
data['dayofweek'] = data.pickup_datetime.dt.dayofweek
data['hour'] = data.pickup_datetime.dt.hour
data = data.drop('pickup_datetime',axis = 1)

#################################################################################

##########################----Model PreProcessing----#############################

###############################################################################

#####Breaking up data into train and test again
#Dropping longitude and latitude
data= data.iloc[:,4:]

"""Checking Multi colinearity""" 
data_vif=calculate_vif_(data)


#Log transformation of variable Distance and Fare 
"""As We have seen above in the density plot of distance and fare amount that our data is 
left skweed. So in order to fix that we have to do log transformation of both the variables."""
data_vif.distance = np.log(data_vif.distance+1)
train.fare_amount = np.log(train.fare_amount+1)


#Breakinf the data again into test and train
X = data_vif.iloc[train['fare_amount'].index,]
test = data_vif.iloc[15450:,:]
y = train.fare_amount

#################################################################################

##########################----Model Building----#############################

###############################################################################

#Linear Regression
X_train,X_test,y_train,y_test=splitter(X,y)
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)


y_pred_train = regressor.predict(X_train)

#Evaluation Metric
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2_sc= r2_score(y_test,y_pred)
rmse_train=np.sqrt(mean_squared_error(y_train,y_pred_train))
r2_sc_train = r2_score(y_train,y_pred_train)
coeff_df = pd.DataFrame(regressor.coef_,X.columns,columns=['Coefficient'])

print("RMSE Score for test is : ", rmse)
print("R^2 Score for test is :", r2_sc)
print("RMSE Score for train is : ",rmse_train )
print("R^2 Score for train is :", r2_sc_train)
coeff_df
 #                Coefficient
#passenger_count     0.004074
#distance            0.768508
#month               0.004535
#day                 0.000090
#dayofweek          -0.003499
#hour                0.000440
"""As expected we can see that the most weightage is of variables distance"""

###############################################################################
#Polynomial Regression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

poly_reg = PolynomialFeatures(degree = 4)
X = poly_reg.fit_transform(X)
poly_reg.fit(X,y)
X_train,X_test,y_train,y_test=splitter(X,y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_train, y_train)
y_pred = lin_reg_2.predict(X_test)


y_pred_train = regressor.predict(X_train)

#Evaluation Metric
rmse=np.sqrt(mean_squared_error(y_test,y_pred)) 
r2_sc= r2_score(y_test,y_pred)
rmse_train=np.sqrt(mean_squared_error(y_train,y_pred_train))
r2_sc_train = r2_score(y_train,y_pred_train)


print("RMSE Score for test is : ", rmse)#0.24
print("R^2 Score for test is :", r2_sc)#0.79
print("RMSE Score for train is : ",rmse_train )#0.25
print("R^2 Score for train is :", r2_sc_train)#0.78



###############################################################################
#Decision Tree
X = data_vif.iloc[train['fare_amount'].index,]
y = train.fare_amount
X_train,X_test,y_train,y_test=splitter(X,y)
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)


y_pred_train = regressor.predict(X_train)

#Evaluation Metric
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2_sc= r2_score(y_test,y_pred)
rmse_train=np.sqrt(mean_squared_error(y_train,y_pred_train))
r2_sc_train = r2_score(y_train,y_pred_train)


print("RMSE Score for test is : ", rmse)
print("R^2 Score for test is :", r2_sc)
print("RMSE Score for train is : ",rmse_train )
print("R^2 Score for train is :", r2_sc_train)



###############################################################################
#Random Forest
X_train,X_test,y_train,y_test=splitter(X,y)
regressor = RandomForestRegressor(n_estimators=200)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)


y_pred_train = regressor.predict(X_train)

#Evaluation Metric
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2_sc= r2_score(y_test,y_pred)
rmse_train=np.sqrt(mean_squared_error(y_train,y_pred_train))
r2_sc_train = r2_score(y_train,y_pred_train)


print("RMSE Score for test is : ", rmse)#0.24
print("R^2 Score for test is :", r2_sc)#0.78
print("RMSE Score for train is : ",rmse_train )#0.099
print("R^2 Score for train is :", r2_sc_train)#0.96


###############################################################################
#KNN
X_sc = StandardScaler()
X = X_sc.fit_transform(X)
y_sc = StandardScaler()
y=np.array(y).reshape(-1,1)
y = y_sc.fit_transform(y)
X_train,X_test,y_train,y_test=splitter(X,y)
regressor = KNeighborsRegressor(n_neighbors=5)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)

#inverse transform
y_pred = y_sc.inverse_transform(y_pred)
y_test = y_sc.inverse_transform(y_test)
y_train = y_sc.inverse_transform(y_train)
y_pred_train = y_sc.inverse_transform(y_pred_train)


#Evaluation Metric
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2_sc= r2_score(y_test,y_pred)
rmse_train=np.sqrt(mean_squared_error(y_train,y_pred_train))
r2_sc_train = r2_score(y_train,y_pred_train)


print("RMSE Score for test is : ", rmse)
print("R^2 Score for test is :", r2_sc)
print("RMSE Score for train is : ",rmse_train )
print("R^2 Score for train is :", r2_sc_train)

###############################################################################
#XGBOOST
X = data_vif.iloc[train['fare_amount'].index,]
y = train.fare_amount
X_train,X_test,y_train,y_test=splitter(X,y)
regressor = XGBRegressor(objective='reg:squarederror',eval_metric='rmse')
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

y_pred_train = regressor.predict(X_train)

#Evaluation Metric
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2_sc= r2_score(y_test,y_pred)
rmse_train=np.sqrt(mean_squared_error(y_train,y_pred_train))
r2_sc_train = r2_score(y_train,y_pred_train)


print("RMSE Score for test is : ", rmse)#0.26
print("R^2 Score for test is :", r2_sc)#0.80
print("RMSE Score for train is : ",rmse_train )#0.24
print("R^2 Score for train is :", r2_sc_train)#0.80

###############################################################################
#Hyperparameter tuning
"""hyperparameter tuning is choosing a set of optimal hyperparameters for a learning algorithm.
A hyperparameter is a parameter whose value is set before the learning process begins. 
We will do hyperparameter tuning in two models Random Forest and XGBOOST. Also we will use try 
both typer of hyper parameter tuning i.e. Random Search CV and Grid Search CV"""
#Grid Search on Random Forest
"""Defining parameters"""
n_estimators = list(range(20,35,1))
depth = list(range(5,14,2))

param_grid = dict(n_estimators = n_estimators,
                  max_depth=depth)

"""Creating Grid dearch"""
grid = GridSearchCV(estimator = RandomForestRegressor(random_state=0),
                    param_grid = param_grid,
                    cv = 5)

grid_result = grid.fit(X_train,y_train)
y_pred = grid_result.predict(X_test)
y_pred_train = regressor.predict(X_train)

"""Cehcking Best parameter"""
print("Best Parameters are: ", grid_result.best_params_)

#Evaluation Metric
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2_sc= r2_score(y_test,y_pred)
rmse_train=np.sqrt(mean_squared_error(y_train,y_pred_train))
r2_sc_train = r2_score(y_train,y_pred_train)


print("RMSE Score for test is : ", rmse)
print("R^2 Score for test is :", r2_sc)
print("RMSE Score for train is : ",rmse_train)
print("R^2 Score for train is :", r2_sc_train)

###############################################################################
#Grid Search on XGBOOST
"""Defining parameters"""
n_estimators = list(range(45,100,1))
depth = list(range(5,15,2))

param_grid = dict(n_estimators = n_estimators,
                  max_depth=depth)

"""Creating Grid dearch"""
grid = GridSearchCV(estimator = XGBRegressor(objective='reg:squarederror',eval_metric='rmse') ,
                    param_grid = param_grid,
                    cv = 5)

grid_result = grid.fit(X_train,y_train)
y_pred = grid_result.predict(X_test)
y_pred_train = grid_result.predict(X_train)

"""Checking Best parameter"""
print("Best Parameters are: ", grid_result.best_params_)
#Best Parameters are:  {'max_depth': 5, 'n_estimators': 49}

#Evaluation Metric
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2_sc= r2_score(y_test,y_pred)
rmse_train=np.sqrt(mean_squared_error(y_train,y_pred_train))
r2_sc_train = r2_score(y_train,y_pred_train)


print("RMSE Score for test is : ", rmse)
print("R^2 Score for test is :", r2_sc)
print("RMSE Score for train is : ",rmse_train)
print("R^2 Score for train is :", r2_sc_train )

###############################################################################

#Random Search on Random Forest
"""Defining parameters"""
n_estimators = list(range(20,100,1))
depth = list(range(1,30,2))

param_random = dict(n_estimators = n_estimators,
                  max_depth=depth)

"""Creating Random search"""
random = RandomizedSearchCV(estimator = RandomForestRegressor(random_state=0),
                    param_distributions = param_random,
                    cv = 5,
                    n_iter = 5)

random_result = random.fit(X_train,y_train)
y_pred = random_result.predict(X_test)
y_pred_train = random_result.predict(X_train)


"""Checking Best parameter"""
print("Best Parameters are: ", random_result.best_params_)
#Best Parameters are:  {'n_estimators': 82, 'max_depth': 7}

#Evaluation Metric
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2_sc= r2_score(y_test,y_pred)
rmse_train=np.sqrt(mean_squared_error(y_train,y_pred_train))
r2_sc_train = r2_score(y_train,y_pred_train)


print("RMSE Score for test is : ", rmse)
print("R^2 Score for test is :", r2_sc)
print("RMSE Score for train is : ",rmse_train)
print("R^2 Score for train is :", r2_sc_train )

###############################################################################

#Random Search on XGBOOST
"""Defining parameters"""
n_estimators = list(range(20,100,1))
depth = list(range(5,100,1))

param_random = dict(n_estimators = n_estimators,
                  max_depth=depth)

"""Creating Random earch"""
random = RandomizedSearchCV(estimator = XGBRegressor(objective='reg:squarederror',eval_metric='rmse') ,
                    param_distributions = param_random,
                    cv = 5,
                    n_iter = 5)

random_result = random.fit(X_train,y_train)
y_pred = random_result.predict(X_test)
y_pred_train = random_result.predict(X_train)


"""Checking Best parameter"""
print("Best Parameters are: ", random_result.best_params_)
#Best Parameters are:  {'n_estimators': 53, 'max_depth': 61}
#Evaluation Metric
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2_sc= r2_score(y_test,y_pred)
rmse_train=np.sqrt(mean_squared_error(y_train,y_pred_train))
r2_sc_train = r2_score(y_train,y_pred_train)


print("RMSE Score for test is : ", rmse)
print("R^2 Score for test is :", r2_sc)
print("RMSE Score for train is : ",rmse_train)
print("R^2 Score for train is :", r2_sc_train )

###############################################################################

######################--Model Selection and Predicting Test case--#############

###############################################################################

"""We have choosed XGBOOST to be the best model and we will predict the test case
with this model only."""


"""Uncomment this

#Building Model again
X = data_vif.iloc[train['fare_amount'].index,]
y = train.fare_amount
X_train,X_test,y_train,y_test=splitter(X,y)
regressor = XGBRegressor(objective='reg:squarederror',eval_metric='rmse')
regressor.fit(X_train,y_train)

#Predicting Test case
y_pred_test = regressor.predict(test)

#Adding prediction column to test file
test = pd.read_csv("test.csv")
test['predicted_fare'] = y_pred_test

#Writting the file
test.to_csv('submission.csv',header=True,index=False)


"""

"""############################################################################################################"""

