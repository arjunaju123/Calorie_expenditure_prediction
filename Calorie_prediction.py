
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
#from xgboost import XGBClassifier
from sklearn import metrics

# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.svm import SVC
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import KNeighborsRegressor

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

# find maximum value of a single column 'x'
#maxClm = calories_data['Calories'].max()

# Choose entries with calories 314.0
df_new = calories_data[calories_data['Calories'] == 314.0]
print(df_new)


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
# plotting the gender column in count plot
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
# model=KNeighborsClassifier()
# model=KNeighborsRegressor
# model1 = KNeighborsRegressor(n_neighbors=2)
print(model)

# training the model with X_train and Y_train
model.fit(X_train, Y_train)

print(model)
test_data_prediction = model.predict(X_test)

# X_test.head()

#Get a new Data and Check the calorie expenditure(in the terminal)
# New_gender=int(input('Enter the gender : '))
# New_age=int(input('Enter the age : '))
# New_height=int(input('Enter the height : '))
# New_weight=int(input('Enter the weight : '))
# New_duration=int(input('Enter the duration : '))
# New_heartrate=int(input('Enter the heart rate : '))
# New_temp=int(input('Enter the body temp : '))
# New_Data=np.array([ [New_gender,New_age,New_height,New_weight,New_duration,New_heartrate,New_temp]])
# New_Data_User = pd.DataFrame(New_Data,columns=['Gender', 'Age','Height','Weight','Duration','Heart_Rate','Body_Temp'])
# print(New_Data_User.head())
# calorie_pred = model.predict(New_Data_User)
# calorie_pred = model.predict(New_Data)
# print(calorie_pred)


# print(model.predict(np.array([[Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp]])))
print(model.predict(np.array([[1,41,172.0,74.0,24.0,98.0,40.8]])))
print(model.predict(np.array([[1,75,199.0,103.0,28.0,123.0,40.5]])))

mae = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("Mean Absolute Error = ", mae)
print(model)

# Saving the xgboost model as a pickle file
import pickle
pickle_file = open("xgboost_model.pkl","wb")
pickle.dump(model, pickle_file)
pickle_file.close()



