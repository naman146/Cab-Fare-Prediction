#########################################        Python             ########################################
Note:-  there maybe a case of depreciation warnings that may popup due to some changes in the default 
	functions of the libraries. And maybe some future warnings may appear. But that will not hinder
	the performance of the code and the code will run fine as normal. So kindly ignore those warnings.

Need to install all the below mentioned libraries of specific functions with their dependencies:
Python 3.6.8
Librarry os (inbuild)
Library pandas 0.24.2
Library numpy 1.16.4
Library matplotlib 3.1.1 
Library seaborn 0.9.0
Library sklearn 0.21.2
Library scipy 1.2.1
Library geopy 1.20.0
Library statsmodels 0.10.0
Library xgboost 0.90
Library folium 0.10.0

1. There are two files "py_cab_fare_gui.py" which needs to be run in an IDE or a GUI like spyder and nother file is
	"py_cab_fare_cli.py" which needs to be run in cli window like DOS. If "py_cab_fare_cli.py" is choosed to be run then
	the script output should be redirected to an output file (python __directory of py file__ > outputfile.txt) in order to get
	proper understanding of the output. The cli python file doesnot include any visualization code. If visualisation is
	needed then only "py_cab_fare_gui.py" file should be run in a GUI like spyder. Kindly Run both the files where the dataset "train.csv"
	and "test.csv" are present with the same file name
NOTE:- Kindly change the working directory in both the code files as required.

#########################################        R            ########################################
Need to install all the below mentioned libraries of specific functions with their dependencies:
R version 3.5.1
Librarry class_7.3-15 
Librarry randomForest_4.6-14
Librarry rpart_4.1-15 
Librarry caret_6.0-80            
Librarry dplyr_0.7.6         
Librarry ggplot2 3.0.0       
library  plyr 1.8.4 
library  DataCombine 0.2.21  
library  leaflet 2.0.1
library  leaflet.extras _1.0.0
library  tidyr 0.8.1
library  lubridate 1.7.4 
library  geosphere 1.5-10 
library  caret  6.0-80 
library  rpart 4.1-15
library  mlr 2.16.0 
library  gbm 2.1.5
library  caTools 1.17.1.2

1. There are two files "R_cab_fare_gui.R" which needs to be run in an IDE or a GUI like R_studio and another file is
	"R_cab_fare_cli.R" which needs to be run in cli window like DOS. If "R_cab_fare_cli.R" is choosed to be run then
	the script output should be redirected to an output file (__R.exe directory__ __directory of R file__) or from cmd if 
	R is in enviorment variable then "R CMD BATCH script.R" in order to get	proper understanding of the output. The cli R
	file doesnot include any visualization code. If visualisation is needed the only "R_cab_fare_gui.R" file should be run in a GUI
	like R_studio. Kindly Run both the files where the dataset "train.csv" and "test.csv" are present with the same file name
NOTE:- Kindly change the working directory in both the code files as required.


#########################################     Submission File          ########################################

submission.csv is the submission file which contains all the predicted values of the test cases in the column "predicted_fare"
