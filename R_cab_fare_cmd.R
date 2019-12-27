rm(list = ls())
setwd("D:/MY PROGRAMMING DATA/edwisor/Project02")

#################################################################################

#Libraries to import
library(plyr) #data manipulation
library(dplyr)# data maniputation
library(DataCombine) # data maniputation
library(ggplot2) #visualization
library(leaflet) #map visualization
library(leaflet.extras) #map visualization
library(tidyr) #data wrangling
library(lubridate) #for date and time
library(geosphere) # for geo distance calculation
library(caret) 
library(rpart) #Decission Tree
library(mlr) 
library(randomForest) #random forest
library(gbm) #XGBoost
library(caTools) #Polynomial
library(class) #KNN

#################################################################################

#################################################################################
"Importing dataset"
train = data.frame(read.csv('train_cab.csv')) 
test = data.frame(read.csv('test.csv')) 
#################################################################################

#################################################################################
"""Exploring dataset"""
#checking head of data
head(train,10)
head(test)

#checking structure of data
str(train)
str(test)

summary(train)
glimpse(train)
#Observations: 16,067
#Variables: 7

summary(test)
glimpse(test)
#Observations: 9,914
#Variables: 6

#unique value of each count
apply(train, 2,function(x) length(table(x))) 

#Changing pickup_date time to datetime object
train$pickup_datetime <- gsub('\\ UTC','',train$pickup_datetime) 
test$pickup_datetime <- gsub('\\ UTC','',test$pickup_datetime) 


#checking NaN values
#for Train data
sum(is.na(train))
apply(train, 2, function(x) {sum(is.na(x))}) # in R, 1 = Row & 2 = Col 
#fare_amount   pickup_datetime  pickup_longitude   pickup_latitude dropoff_longitude  dropoff_latitude   passenger_count 
#0                 0                 0                 0                 0                 0                55 
train = DropNA(train)

#for test data
apply(test, 2, function(x) {sum(is.na(x))})
#pickup_datetime  pickup_longitude   pickup_latitude dropoff_longitude  dropoff_latitude   passenger_count 
#0                 0                 0                 0                 0                 0 

#so we dont have NA value in test and can move forward easily

"changing fare_amount in float and dropping negative values"
train = train[-c(2487,2040,13033),]

#creating a small dataframe for replacing null values in fare amount

replace_df = data.frame(from = "", to = 15.04)
train = FindReplace(data=train,Var="fare_amount", replaceData=replace_df,from="from",to="to")
train$fare_amount = as.numeric(train$fare_amount)
train = DropNA(train)
train$fare_amount = as.numeric(train$fare_amount)


#lets see sort passenger count varibale into descending order and then see what are those high values
head(train[order(-train$passenger_count),])
head(train[order(train$passenger_count),])
#So we go few values ranging from 0 to 5345 which is not possible for a cab service
#So we need to replace these outliers
train$passenger_count = as.integer(train$passenger_count)
train$passenger_count[train$passenger_count<1] <- NA 
train$passenger_count[train$passenger_count>6] <- NA
train <- DropNA(train)
train$passenger_count = as.factor(train$passenger_count)



#Now lets see Pickup_datetime
rownames(train) = NULL
train = train[-1279,]
rownames(train) = NULL


#train Longitude and Latitude
train = train[-c(6928,1135,5783,5606),]
rownames(train) = NULL
train$pickup_latitude[train$pickup_latitude==0] <- NA
train$dropoff_longitude[train$dropoff_longitude==0] <- NA
train = DropNA(train)
rownames(train) = NULL


xyz = train[round(train['pickup_longitude']) == 41,]
train[rownames(xyz),c('pickup_longitude','pickup_latitude')]=train[rownames(xyz),c('pickup_latitude','pickup_longitude')]
train[rownames(xyz),c('dropoff_longitude','dropoff_latitude')]=train[rownames(xyz),c('dropoff_latitude','dropoff_longitude')]
rm(xyz)


data = rbind(train[,c('pickup_datetime', 'pickup_longitude', 'pickup_latitude',
                      'dropoff_longitude', 'dropoff_latitude', 'passenger_count')],
             test)
rownames(data) = NULL

data$month = month(data$pickup_datetime)
data$hour = hour(data$pickup_datetime)
data$dayofweek = wday(data$pickup_datetime)
data$year = year(data$pickup_datetime)
data$day = mday(data$pickup_datetime)
data = DropNA(data)


train$fare_amount[c(563,915,1005,1055,1259,1401,950,1006)] = 10.4 



#################################################################################

##########################----Feature Enginnering---#############################

###############################################################################
rownames(data) = NULL
rownames(train) = NULL
data = data[-5488,]
train = train[-5488,]
rownames(data) = NULL
rownames(train) = NULL



#Creating Column of distance from pickup and dropoff
for (row in 1:nrow(data)){
  data$distance[row]=distm(c(data$pickup_longitude[row], data$pickup_latitude[row]),
                           c(data$dropoff_longitude[row], data$pickup_latitude[row]), 
                           fun = distHaversine) 
}



data$distance = data$distance/1000
data = data[-5660,]
train = train[-5660,]
rownames(data) = NULL
rownames(train) = NULL


#the amount is very less, all other values has=ve linear relation. So we need to remove these outliers

data = data[,6:12]

#Breaking the data again into test and train
cont = c( 'distance') 
cata = c("passenger_count" ,"day", "month","hour","dayofweek","year" )
data$day = as.factor(data$day)
data$month = as.factor(data$month)
data$hour = as.factor(data$hour)
data$dayofweek = as.factor(data$dayofweek)
data$year = as.factor(data$year)
data$passenger_count = as.factor(data$passenger_count)

train01 = data[rownames(train),]
train01$fare_amount = train$fare_amount
train = train01
rm(train01)

test = data[nrow(train):25514,]


#dropping Distance with value 0
train$distance[train$distance==0] <- NA
train = DropNA(train)
rownames(train) = NULL


#Feature Scaling

signedlog10 = function(x) { 
  ifelse(abs(x) <= 1, 0, sign(x)*log10(abs(x))) 
} 

train$fare_amount = signedlog10(train$fare_amount) 
train$distance = signedlog10(train$distance) 
test$distance = signedlog10(test$distance) 

##checking distribution 
hist(train$fare_amount) 
hist(train$distance)


#Normalization 
cont = c( 'distance') 
for(i in cont) 
{ 
  print(i) 
  train[,i] = (train[,i] - min(train[,i]))/(max(train[,i])-min(train[,i])) 
  test[,i] = (test[,i] - min(test[,i]))/(max(test[,i])-min(test[,i]))
} 



cata = c("passenger_count" ,"day", "month","hour","dayofweek","year" )

#Creating dummy variables for categorical variables 
train =  createDummyFeatures(train, cata)
test =  createDummyFeatures(test, cata)





################################################################ 
# Sampling of Data # 
################################################################ 

# #Divide data into trainset and testset using stratified sampling method 


set.seed(1234)
train_index = createDataPartition(train$fare_amount,p=0.8,list = FALSE)
X_train = train[train_index,]
X_test = train[-train_index,]



#################################################################################

##########################----Model Building----#############################

###############################################################################

rSquared = function(actual,preds){
  rss <- sum((preds - actual) ^ 2)  ## residual sum of squares
  tss <- sum((actual - mean(actual)) ^ 2)  ## total sum of squares
  1 - rss/tss
  
}


#Linear Regression


#Develop Model on training data 
fit_LR = lm(formula = fare_amount ~ ., data = X_train) 


#Lets predict for test data 
pred_LR_test = predict(fit_LR, newdata =X_test) 


# For test data 
print(postResample(pred = pred_LR_test, obs = X_test$fare_amount)) 
#RMSE         Rsquared       MAE 
#0.1622500   0.6217942  0.1192584 

#Compute R^2 

lr_r2 = rSquared(X_test$fare_amount,pred_LR_test) 
print(lr_r2) 

#Compute MSE 
lr_mse = mean((X_test$fare_amount - pred_LR_test)^2) 
print(lr_mse)
print(sqrt(lr_mse))


#####################################################################
#Decision Tree
#Develop Model on training data 
fit_DT = rpart(fare_amount ~., data = X_train, method = "anova") 


#Variable importance 
fit_DT$variable.importance 
#distance 
#451.8346

#Lets predict for test data 
pred_DT_test = predict(fit_DT, X_test) 

# For test data 
print(postResample(pred = pred_DT_test, obs = X_test$fare_amount)) 
#     RMSE  Rsquared       MAE 
#0.1732958 0.5682772 0.131113

#Compute R^2 
dt_r2 = rSquared(X_test$fare_amount,pred_DT_test) 
print(dt_r2) 

#Compute MSE 
dt_mse = mean((X_test$fare_amount - pred_DT_test)^2) 
print(dt_mse) 

#####################################################################
#Random Forest

#Develop Model on training data 
fit_RF = randomForest(fare_amount~., data = X_train) 


#Lets predict for test data 
pred_RF_test = predict(fit_RF, X_test) 
# For test data 
print(postResample(pred = pred_RF_test, obs = X_test$fare_amount)) 
#RMSE  Rsquared       MAE 
#0.1701062 0.6091653 0.1293937 

#Compute R^2 
rf_r2 = rSquared(X_test$fare_amount, pred_RF_test) 
print(rf_r2) 

#Compute MSE 
rf_mse = mean((X_test$fare_amount - pred_RF_test)^2) 
print(rf_mse) 


#####################################################################
#XGBOOST
#Develop Model on training data 
fit_XGB = gbm(fare_amount~., data = X_train, n.trees = 500, interaction.depth = 2) 


#Lets predict for test data 
pred_XGB_test = predict(fit_XGB, X_test, n.trees = 500) 


# For test data 
print(postResample(pred = pred_XGB_test, obs = X_test$fare_amount)) 
#     RMSE  Rsquared       MAE 
#0.1622140 0.6222086 0.1198040 

#Compute R^2 
xgb_r2 = rSquared(X_test$fare_amount, pred_XGB_test) 
print(xgb_r2) 

#Compute MSE 
xgb_mse = mean((X_test$fare_amount - pred_XGB_test)^2) 
print(xgb_mse) 

#####################################################################
#Polynimial Regression

# Fitting Linear Regression to the dataset
lin_reg_pl = lm(fare_amount~., data = X_train)

# Fitting Polynomial Regression to the dataset
features = colnames(X_train[7])

for(feature in features){
  X_train[paste(feature,"_sq",sep = '')] = X_train[feature]^2
  X_train[paste(feature,"_cu",sep = '')] = X_train[feature]^3
  X_train[paste(feature,"_qu",sep = '')] = X_train[feature]^4
  X_test[paste(feature,"_sq",sep = '')] = X_test[feature]^2
  X_test[paste(feature,"_cu",sep = '')] = X_test[feature]^3
  X_test[paste(feature,"_qu",sep = '')] = X_test[feature]^4
}

poly_reg = lm(fare_amount~., data = X_train)
pred_poly_reg=predict(poly_reg, newdata = X_test)


# For test data 
print(postResample(pred = pred_poly_reg, obs = X_test$fare_amount)) 


#Compute R^2 
poly_r2 = rSquared(X_test$fare_amount, pred_poly_reg) 
print(poly_r2) 

#Compute MSE 
poly_mse = mean((X_test$fare_amount - pred_poly_reg)^2) 
print(poly_mse) 

#####################################################################
#Creating Submission file
#set.seed(1234)
#train_index = createDataPartition(train$fare_amount,p=0.8,list = FALSE)
#X_train = train[train_index,]
#X_test = train[-train_index,]
#fit_XGB = gbm(fare_amount~., data = X_train, n.trees = 500, interaction.depth = 5) 
#pred_XGB_test = predict(fit_XGB, test, n.trees = 500) 
#Prediction
#test$predicted_fare = pred_XGB_test
#write.csv(test, file = "submission01.csv",row.names=FALSE)