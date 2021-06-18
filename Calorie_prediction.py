
#Importing the Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# I personally tend to go with the ensemble model like XG Boost and  CART also.
# (especially since they tend to increase prediction accuracy by combining the predictions from multiple models together).
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Data Collection & Processing
#
# loading the data from csv file to a Pandas DataFrame
calories = pd.read_csv('calories.csv')

# print the first 5 rows of the dataframe
print(calories.head())

exercise_data = pd.read_csv('exercise.csv')

print(exercise_data.head())

#Combining the two Dataframes
calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

# #Converting the text data to numerical values
#
calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)
# print(calories_data.head())
#Lean Body Mass Formula for Adults
# The Boer Formula:1
#
# For males:
# eLBM = 0.407W + 0.267H - 19.2
# For females:
# eLBM = 0.252W + 0.473H - 48.3

#Adding additional feature "lean body mass".
#Body Composition: A person with more muscle will often burn more calories than a person with higher body fat.
#the more Lean Body Mass you have, the greater your  Metabolic Rate will be.
calories_data['Lean_Body_mass']=(calories_data['Weight']*0.407+calories_data['Height']*0.267-19.2)*calories_data['Gender']

calories_data.loc[calories_data['Lean_Body_mass'] == 0, 'Lean_Body_mass'] =calories_data['Weight'] * 0.252 + calories_data['Height'] * 0.473 - 48.3

print(calories_data)
#checking the number of rows and columns
print(calories_data.shape)

# getting some informations about the data
print(calories_data.info())

# checking for missing values
print(calories_data.isnull().sum())

# Data Analysis
# get some statistical measures about the data
print(calories_data.describe())

# find the maximum of each column
maxValues = calories_data.max()
print(maxValues)

#Data Visualization

# sns.set_style('whitegrid')
# sns.lmplot(x ='Duration', y ='Calories', data = calories_data)
# plt.show()
# sns.lmplot(x ='Lean_Body_mass', y ='Calories', data = calories_data)
# plt.show()
# # sns.lmplot(x ='Age', y ='Calories', data = calories_data)
# # sns.lmplot(x ='Height', y ='Calories', data = calories_data)
# # sns.lmplot(x ='Weight', y ='Calories', data = calories_data)
# # sns.lmplot(x ='Heart_Rate', y ='Calories', data = calories_data)
# # sns.lmplot(x ='Body_Temp', y ='Calories', data = calories_data)
# # a_plot=sns.lmplot(x ='Duration', y ='Calories', data = calories_data)
# # a_plot.set(xlim=(0, 60))
# # a_plot.set(ylim=(0, 800))
# # plt.show()
#
# # sns.set()
# # plotting the gender column in count
# # male and female are almost equally distributed.Therefore good distribution.
# # count plot for categorical data only.
# # sns.countplot(calories_data['Gender'])
#
# # finding the distribution of "Age" column
#
# #sns.distplot(calories_data['Age'])
# #plt.show()
# # finding the distribution of "Height" column
# # sns.distplot(calories_data['Height'])
# # plt.show()
# # finding the distribution of "Lean body mass" column
# sns.distplot(calories_data['Lean_Body_mass'])
# plt.show()
# # finding the distribution of "Weight" column
# # sns.distplot(calories_data['Weight'])
# # plt.show()
# # sns.distplot(calories_data['Duration'])
# # plt.show()
# # sns.distplot(calories_data['Heart_Rate'])
# # plt.show()
# # sns.distplot(calories_data['Body_Temp'])
# # plt.show()
#
# #A correlation between variables indicates that as one variable changes in value, the other variable tends to change in a specific direction.
# # Understanding that relationship is useful because we can use the value of one variable to predict the value of the other variable.
# #positive correlation - If one feature increses other feature also increases
# #negative correlation - If one feature increases other feature decreases
# correlation = calories_data.corr()
# # constructing a heatmap to understand the correlation
#
# # plt.figure(figsize=(10,10))
# # sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
# # plt.show()
#

#
#
X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
#X = calories_data.drop(columns=['Calories'], axis=1)
Y = calories_data['Calories']

#print(X)

#print(Y)
#
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape,Y_train.shape,Y_test.shape)

#LINEAR REGRESSION MODEL
#Here we are going to use Linear Regression model and then check r2_score.
# If r2_score is more then Linear else not linear.
# from sklearn.metrics import r2_score
# # loading the model
# model = LinearRegression()
# # training the model with X_train and Y_train
# model.fit(X_train, Y_train)
# test_data_prediction = model.predict(X_test)
# print('R2 Score is: ', r2_score(Y_test, test_data_prediction))
# #Actually r2_score defined the accuracy. So if accuracy is very low then its denote linear model is not fit for this data.
# #Evaluation
# #Prediction on Test Data
# mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
# print("Mean Absolute Error of linear regression model = ", mae)
# print(test_data_prediction)
# #

# # SUPPORT VECTOR REGRESSION MODEL
# # loading the model
# model=SVR()
# # training the model with X_train
# model.fit(X_train, Y_train)
# test_data_prediction = model.predict(X_test)
# #Evaluation
# #Prediction on Test Data
# mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
# print("Mean Absolute Error of Support vector regressor is = ", mae)
# print(test_data_prediction)
# #

# #KNeighbours REGRESSION MODEL
# # loading the model
# model = KNeighborsRegressor(n_neighbors=2)
# # training the model with X_train
# model.fit(X_train, Y_train)
# test_data_prediction = model.predict(X_test)
# #Evaluation
# #Prediction on Test Data
# mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
# print("Mean Absolute Error of K neighbours regressor is = ", mae)
# print(test_data_prediction)
# #

# #DECISION TREE MODEL
# # loading the model
# model=DecisionTreeRegressor()
# # training the model with X_train
# model.fit(X_train, Y_train)
# test_data_prediction = model.predict(X_test)
# #Evaluation
# #Prediction on Test Data
# mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
# print("Mean Absolute Error of Decision tree model is = ", mae)
# print(test_data_prediction)
#

# XGBOOST REGRESSION MODEL
# loading the model
model = XGBRegressor()
# training the model with X_train
model.fit(X_train, Y_train)
test_data_prediction = model.predict(X_test)
#Evaluation
#Prediction on Test Data
mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
print("Mean Absolute Error of XGBOOST Regression model is = ", mae)
print(test_data_prediction)

#print(model.predict(np.array([[1,21,182,87.0,30.0,98.0,40.8]])))


# Saving the xgboost model as a pickle file
import pickle
pickle_file = open("xgboost_model.pkl","wb")
pickle.dump(model, pickle_file)
pickle_file.close()



