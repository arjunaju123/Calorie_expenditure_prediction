import numpy as np
import streamlit as st
import pickle
import time

st.write("""
# Calorie expenditure prediction app
 """)
#Moderate exercise intensity: 50% to about 70% of your maximum heart rate
st.markdown(f'<p style="color:#FF0000;font-size:20px;border-radius:2%;">THIS APP CALCULATES THE CALORIC EXPENDITURE FOR A MODERATE INTENSITY EXERCISE(HEART BEAT AROUND 130 BEATS PER MINUTE) UNDER A DURATION OF 30 MIN.</p>',unsafe_allow_html=True)
cal_predi = open('xgboost_model.pkl', 'rb')
predictor = pickle.load(cal_predi)

print(predictor)
#Text Input
Gender = st.selectbox('Enter your Gender(Male or Female)',('Male', 'Female'))

if(Gender=='Male'):
    Gender=1
else:
    Gender=0

Age =st.number_input("Enter the Age",step=1,max_value=80)
Height = st.selectbox('Choose the metric to input height',('In feet', 'In metres','In centimetres'))
Height_cm=0
if(Height=='In feet' or Height=='In metres' or Height=='In centimetres' ):
    if(Height=='In feet'):
        Height_feet = st.number_input("Enter your height in feet", step=0.1,max_value=7.283)
        Height_cm=Height_feet*30.48
    elif(Height=='In metres'):
        Height_metre = st.number_input("Enter your height in metre", step=0.1, max_value=2.22)
        Height_cm = Height_metre * 100
    else:
        Height_cm=st.number_input("Enter the Height in centimetres:",step=0.1,max_value=222.0)
#Height = st.number_input("Enter the Height:",step=0.1,max_value=222.0)

Weight_Kg=0
#Weight_pounds=0
Weight = st.selectbox('Choose the metric to input body weight',('In Kilograms', 'In Pounds'))
if(Weight=='In Kilograms' or Weight=='In Pounds'):
    #Weight_Kg = 0
    if(Weight=='In Kilograms'):
        Weight_Kg = st.number_input("Enter your Body Weight in kilograms", step=0.1)
    elif(Weight=='In Pounds'):
        Weight_pounds = st.number_input("Enter your Body Weight in pounds", step=0.1)
        Weight_Kg = Weight_pounds/2.2

#Weight =  st.number_input("Enter the Weight",step=0.1)
Lean_Body_Mass=st.number_input("Enter your lean body mass",step=0.1)
Duration =  st.number_input("Enter the Duration of physical activity in minutes",step=0.1,max_value=30.0)
Heart_Rate =  st.number_input("Enter the average Heart beat rate per minute",step=1,max_value=130)
#Body_Temp = st.number_input("Enter the average body temperature:",step=0.1,max_value=41.0)
Body_Temp_cel=0
Body_Temp = st.selectbox('Choose the metric to input the Body Temperature',('In Celsius', 'In Faranheit'))
if(Body_Temp=='In Celsius' or Body_Temp=='In Faranheit'):
    if(Body_Temp=='In Celsius'):
        Body_Temp_cel = st.number_input("Enter your Body temperature in celsius", step=0.1,max_value=41.0)
    elif(Body_Temp=='In Faranheit'):
        Body_Temp_far = st.number_input("Enter your Body temperature in faranheit", step=0.1,max_value=105.8)
        Body_Temp_cel = (Body_Temp_far - 32.0) * 5.0/9.0
submit = st.button("Predict")

age=Age*2

#arr=predictor.predict(np.array([[Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp]]))
if submit:
    if (Weight_Kg==0.0 or Body_Temp_cel == 0.0 or Heart_Rate == 0 or Duration==0.0 or Age==0 or Lean_Body_Mass==0.0 or Height_cm==0.0):
        st.warning("Alert!!!! Please Enter all the required fields")
    else:
        arr = predictor.predict(np.array([[Gender, Age, Height_cm, Weight_Kg, Duration, Heart_Rate, Body_Temp_cel,Lean_Body_Mass]]))
        string = str(arr[0])
        string += " Calories are burned"
        with st.spinner('Wait for it...'):
            time.sleep(3)
            st.success(string)


cb = st.checkbox('Click here to calculate your maintainence calories:')
if cb:

    st.write("""
    # MAINTAINENCE CALORIE CALCULATOR
     """)

    bf = st.number_input("Enter your body fat percentage :")

    box = st.checkbox('Click here to get an idea about your body fat percentage:')
    if box:
        st.image("body-fat-percentage-calc.jpg", width=None)


    activity_level = st.selectbox('Enter the activity level',('Very light', 'Light','Moderate','Heavy','Very heavy'))

    activity_level = activity_level.upper()

    lean_factor = 0
    if (Gender == 1):
        Weight_Kg = Weight_Kg * 1
        if (bf >= 10 and bf <= 14):
            lean_factor = 1.0
        elif (bf > 14 and bf <= 20):
            lean_factor = 0.95
        elif (bf > 20 and bf <= 28):
            lean_factor = 0.90
        elif (bf > 28):
            lean_factor = 0.85
    elif (Gender == 0):
        Weight_Kg = Weight_Kg * 0.9
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

    BMR = Weight_Kg * 24 * (lean_factor)

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

        if (Body_Temp_cel == 0.0 or Heart_Rate == 0 or Duration == 0.0 or Age == 0 or Height_cm == 0.0 or Weight_Kg == 0.0):
            st.warning("Alert!!!! Please Enter all the required fields")

        elif(bf==0.0 and activity_level==0):
            st.warning("Alert!!!! Please Enter the body fat percentage and a valid activity level")

        elif(bf==0.0):
            st.warning("Alert!!!! Please Enter the body fat percentage")
        elif(activity_level==0):
            st.warning("Alert!!!! Please Enter a valid Activity level")


        else:
            string=str(main_cal)
            string+=" calories per day is necessary to maintain your current body fat"
            with st.spinner('Wait for it...'):
                time.sleep(3)
                st.success(string)




