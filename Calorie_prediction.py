
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
# loading the data from csv file to a Pandas DataFrame
calories = pd.read_csv('calories.csv')

# print the first 5 rows of the dataframe
#calories.head()

exercise_data = pd.read_csv('exercise.csv')

#exercise_data.head()

calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

#print(calories_data.head())

sns.set_style('whitegrid')
a_plot=sns.lmplot(x ='Duration', y ='Calories', data = calories_data)
a_plot.set(xlim=(0, 60))
a_plot.set(ylim=(0, 800))
plt.show()

# find the maximum of each column
maxValues = calories_data.max()

print(maxValues)

# Choose entries with id p01
df_new = calories_data[calories_data['Calories'] == 314.0]

print(df_new)
# find maximum value of a
# single column 'x'
#maxClm = calories_data['Calories'].max()

# print("Maximum value in column 'x': ")
# print(maxClm)

# checking the number of rows and columns
#calories_data.shape

# getting some informations about the data
#calories_data.info()

# checking for missing values
calories_data.isnull().sum()

# get some statistical measures about the data
#calories_data.describe()

# sns.set()
#
# # plotting the gender column in count plot
# sns.countplot(calories_data['Gender'])
#
# # finding the distribution of "Age" column
# sns.distplot(calories_data['Age'])

# finding the distribution of "Height" column
# sns.distplot(calories_data['Height'])

# finding the distribution of "Weight" column
# sns.distplot(calories_data['Weight'])

correlation = calories_data.corr()

# constructing a heatmap to understand the correlation

plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)

calories_data.head()

X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
Y = calories_data['Calories']

#print(X)

#print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

#print(X.shape, X_train.shape, X_test.shape)

# loading the model
model = XGBRegressor()
# model = XGBClassifier()
# model = LogisticRegression()
# model = MultinomialNB()
# model = DecisionTreeClassifier()
# model=DecisionTreeRegressor()
# model=SVC()
# model=SVR()
#model=KNeighborsClassifier()
# model=KNeighborsRegressor
# model1 = KNeighborsRegressor(n_neighbors=2)
print(model)
# training the model with X_train
# xgboost_model=model.fit(X_train, Y_train)
model.fit(X_train, Y_train)

print(model)
test_data_prediction = model.predict(X_test)

# X_test.head()

#Get a new Data and Check the calorie expenditure
# New_gender=int(input('Enter the gender : '))
# New_age=int(input('Enter the age : '))
# New_height=int(input('Enter the height : '))
# New_weight=int(input('Enter the weight : '))
# New_duration=int(input('Enter the duration : '))
# New_heartrate=int(input('Enter the heart rate : '))
# New_temp=int(input('Enter the body temp : '))
# New_Data=np.array([ [New_gender,New_age,New_height,New_weight,New_duration,New_heartrate,New_temp]])
# # New_Data_User = pd.DataFrame(New_Data,columns=['Gender', 'Age','Height','Weight','Duration','Heart_Rate','Body_Temp'])
# # print(New_Data_User.head())
# # calorie_pred = model.predict(New_Data_User)
# calorie_pred = model.predict(New_Data)
# print(calorie_pred)

#np.array([[1,41,172.0,74.0,24.0,98.0,40.8]])

# test_data_prediction1 = model.predict(np.array([[1,41,172.0,74.0,24.0,98.0,40.8]]))
# print(model.predict(np.array([[Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp]])))
print(model.predict(np.array([[1,41,172.0,74.0,24.0,98.0,40.8]])))
print("Done")
# print(test_data_prediction1)

mae = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("Mean Absolute Error = ", mae)
print(model.predict(np.array([[1,75,199.0,103.0,28.0,123.0,40.5]])))
print(model)
# Saving the multinomial nb model as a pickle file
import pickle
pickle_file = open("xgboost_model.pkl","wb")
pickle.dump(model, pickle_file)
pickle_file.close()

# import xgboost
# print(xgboost.__version__)
# import joblib
# joblib.dump(model, 'xgb_model.joblib.dat')
#joblib.dump(vectorizer, 'CountVectorizer.joblib')

# import joblib
# print("""
# # Calorie prediction app
#  """)
# #xgboost_predictor = open('xgboost_model.pkl','rb')
# predictor = joblib.load('xgb_model.joblib.dat')
# print(predictor)
# #predictor = pickle.load(xgboost_predictor)
# #Text Input
# Gender =  input("Enter the Gender: (1-Male,0-Female)")
# Age = input("Enter the Age",)
# Height = input("Enter the Height:")
# Weight =  input("Enter the Weight")
# Duration =  input("Enter the Duration")
# Heart_Rate =  input("Enter the Heart rate")
# Body_Temp = input("Enter the body temperature:")
#
#
# print("Success")
# print(predictor.predict(np.array([[Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp]])))

import pandas as pd
# import numpy as np
# import nltk
# import streamlit as st
# import pandas as pd
# import pickle
# import joblib
# st.write("""
# # Calorie prediction app
#  """)
# #xgboost_predictor = open('xgboost_model.pkl','rb')
# #predictor = joblib.load('xgb_model.joblib.dat')
# #predictor = pickle.load(xgboost_predictor)
# #print(predictor)
# #Text Input
# Gender =  st.text_input("Enter the Gender: (1-Male,0-Female)")
# Age = st.text_input("Enter the Age",)
# Height = st.text_input("Enter the Height:")
# Weight =  st.text_input("Enter the Weight")
# Duration =  st.text_input("Enter the Duration")
# Heart_Rate =  st.text_input("Enter the Heart rate")
# Body_Temp = st.text_input("Enter the body temperature:")
# submit = st.button("Predict")
#
# st.write(type(Age))
# # arr=[[Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp]]
# # st.write(type(arr))
# #st.write(type(predictor.predict(np.array([[Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp]]))))
# if submit:
#     st.write("Success")
# 	#result = classifier.predict([[Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp]])
#     st.write(model.predict(np.array([[1,41,172.0,74.0,24.0,98.0,40.8]])))
    #st.write(type(predictor.predict([[Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp]])))
    #st.write(type(result))
	# if result ==0:
	# 	st.write("yes")
	# else:
	# 	st.write("No")
	# 	#st.write(result)
# if st.button('Submit'):
#     st.write('The email you entered is:',spam_predict(vectorised_text))