import pandas as pd
import numpy as np
import nltk
import streamlit as st
import pickle
import joblib
import time

st.write("""
# Calorie expenditure prediction app
 """)
cal_predi = open('xgboost_model.pkl', 'rb')
predictor = pickle.load(cal_predi)

print(predictor)
#Text Input
Gender = st.selectbox('Enter your Gender(Male or Female)',('Male', 'Female'))

if(Gender=='Male'):
    Gender=1
else:
    Gender=0
#print(type(Gender))
#st.write('You selected:', option)
#Gender =  st.number_input("Enter the Gender: (1-Male,0-Female)", min_value=0, max_value=1,step=1)
Age =st.number_input("Enter the Age",step=1)
Height = st.number_input("Enter the Height:",step=0.1)
Weight =  st.number_input("Enter the Weight",step=0.1)
Duration =  st.number_input("Enter the Duration of physical activity in minutes",step=0.1)
Heart_Rate =  st.number_input("Enter the average Heart beat rate per minute",step=1)
Body_Temp = st.number_input("Enter the average body temperature:",step=0.1)
submit = st.button("Predict")

age=Age*2
# st.write(age)
#13079051,male,75,199.0,103.0,28.0,123.0,40.5
arr=predictor.predict(np.array([[Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp]]))

#out_arr=arr[0]
#st.text(out_arr)
# st.text(type(out_arr))

if submit:

    if (Body_Temp == 0.0 or Heart_Rate == 0 or Duration==0.0 or Age==0  or Height==0.0 or Weight==0.0):
        st.warning("Alert!!!! Please Enter all the required fields")

    # elif (bf == 0.0):
    #     st.warning("Alert!!!! Please Enter the body fat percentage")
    # elif (activity_level == 0):
    #     st.warning("Alert!!!! Please Enter a valid Activity level")
    #     # st.subheader(main_cal)
    #
    else:
        arr = predictor.predict(np.array([[Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]]))
        string = str(arr[0])
        string += " Calories are burned"
        with st.spinner('Wait for it...'):
            time.sleep(3)
            st.success(string)

        #st.markdown(f'<p style="color:#ffffff;font-size:35px;border-radius:2%;">MAINTAINENCE CALORIE CALCULATOR</p>',unsafe_allow_html=True)
    #st.subheader(string)
    #st.success("calories")

cb = st.checkbox('Click here to calculate your maintainence calories:')
#f'<p style="color:#33ff33;font-size:32px;border-radius:1%;">MAINTAINENCE CALORIE CALCULATOR</p>', unsafe_allow_html=True
if cb:
    #st.markdown(f'<p style="color:#ffffff;font-size:35px;border-radius:2%;">MAINTAINENCE CALORIE CALCULATOR</p>', unsafe_allow_html=True)
    #st.header('MAINTAINENCE CALORIE CALCULATOR')
    st.write("""
    # MAINTAINENCE CALORIE CALCULATOR
     """)
    # if(arr[0]>=0 and arr[0]<=300):
    #     st.subheader("Very light")
    #
    # elif(arr[0]>300 and arr[0]<=500):
    #     st.subheader("light")
    #
    # elif(arr[0]>500 and arr[0]<=800):
    #     st.subheader("moderate")
    #
    # elif(arr[0]>800 and arr[0]<=1000):
    #     st.subheader("Heavy")
    #
    # else:
    #     st.subheader("Very heavy")

    bf = st.number_input("Enter your body fat percentage :")

    box = st.checkbox('Click here to get an idea about your body fat percentage:')
    if box:
        st.image("body-fat-percentage-calc.jpg", width=None)


    activity_level = st.text_input("Enter the activity level(very light,light,moderate,heavy,very heavy)")
    activity_level = activity_level.upper()

    lean_factor = 0
    if (Gender == 1):
        Weight = Weight * 1
        if (bf >= 10 and bf <= 14):
            lean_factor = 1.0
        elif (bf > 14 and bf <= 20):
            lean_factor = 0.95
        elif (bf > 20 and bf <= 28):
            lean_factor = 0.90
        elif (bf > 28):
            lean_factor = 0.85
    elif (Gender == 0):
        Weight = Weight * 0.9
        if (bf >= 14 and bf <= 18):
            lean_factor = 1.0
        elif (bf > 18 and bf <= 28):
            lean_factor = 0.95
        elif (bf > 28 and bf <= 38):
            lean_factor = 0.90
        elif (bf > 38):
            lean_factor = 0.85
    else:
        st.text("Enter a valid input-- 1 for male and 0 for female")

    BMR = Weight * 24 * (lean_factor)

    if (activity_level == 'VERY LIGHT'):
        main_cal = BMR * 1.3

    elif (activity_level == 'LIGHT'):
        main_cal = BMR * 1.55

    elif (activity_level == 'MODERATE'):
        main_cal = BMR * 1.65

    elif (activity_level == 'HEAVY'):
        main_cal = BMR * 1.80

    elif (activity_level == 'VERY HEAVY'):
        main_cal = BMR * 2.00

    else:
        activity_level = 0
    submit = st.button("Submit")
    if submit:
        # st.write("Success")
        #st.write(type(main_cal))
        if (Body_Temp == 0.0 or Heart_Rate == 0 or Duration == 0.0 or Age == 0 or Height == 0.0 or Weight == 0.0):
            st.warning("Alert!!!! Please Enter all the required fields")

        elif(bf==0.0 and activity_level==0):
            st.warning("Alert!!!! Please Enter the body fat percentage and a valid activity level")

        elif(bf==0.0):
            st.warning("Alert!!!! Please Enter the body fat percentage")
        elif(activity_level==0):
            st.warning("Alert!!!! Please Enter a valid Activity level")
        #st.subheader(main_cal)

        else:
            string=str(main_cal)
            string+=" calories per day is necessary to maintain your current body fat"
            with st.spinner('Wait for it...'):
                time.sleep(3)
                st.success(string)


