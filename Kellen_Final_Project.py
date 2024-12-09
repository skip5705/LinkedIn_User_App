#### Set-Up -------------------

# Import Packages
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Read in data
s = pd.read_csv("social_media_usage.csv")

#### Define Functions----------

# clean_sm function: creates a binary variable
def clean_sm(x) : # x may be used on dataframe in format df["column"]
    # Apply condition: if x = 1, then 1, otherwise 0
    x = np.where(x == 1, 1, 0)
    return x

# model1_predict function: takes 6 variables to predict LinkedIn Use
def model1_predict(income, education, parent, married, female, age) :
    person = [income, education, parent, married, female, age]

    pred_class = model1.predict([person])
    pred_prob = model1.predict_proba([person])

    print(f"Predicted class: {pred_class[0]}") # 0 = Not LinkedIn User, 1 = LinkedIn User
    print(f"Probability that this person is a LinkedIn User: {round(pred_prob[0][1], 4)}")

#### Create training dataset---

# Create dataframe "ss"
ss = pd.DataFrame({
    "sm_li":clean_sm(s["web1h"]),
    "income":np.where(s["income"] <= 9, s["income"], np.nan),
    "education":np.where(s["educ2"] <= 8, s["educ2"], np.nan),
    "parent":clean_sm(s["par"]),
    "married":clean_sm(s["marital"]),
    "female":np.where(s["gender"] == 2, 1, 0),
    "age":np.where(s["age"] <= 98, s["age"], np.nan)})

# Drop missing values 
ss = ss.dropna()

#### Train Model --------------

# Create target (y) and feature(s) selection (X)
y = ss["sm_li"]
X = ss[["income", 
        "education", 
        "parent", 
        "married", 
        "female", 
        "age"
       ]]

# Initialize logistic regression model
model1 = LogisticRegression(class_weight = "balanced")

# Fit model to training data
model1.fit(X.values, y)

#### App?

def sent_app(name, user_income, user_education, user_parent, user_married,
             user_female, user_age) :
    try: 
        person = [user_income, user_education, user_parent, user_married,
                  user_female, user_age]   
        
        # Income Level Conversion
        if person[0] == "Less than 10,000":
            person[0] = 1
        elif person[0] == "10,000 to under 20,000":
            person[0] = 2
        elif person[0] == "20,000 to under 30,000":
            person[0] = 3
        elif person[0] == "30,000 to under 40,000":
            person[0] = 4
        elif person[0] == "40,000 to under 50,000":
            person[0] = 5
        elif person[0] == "50,000 to under 75,000":
            person[0] = 6
        elif person[0] == "75,000 to under 100,000":
            person[0] = 7
        elif person[0] == "100,000 to under 150,000":
            person[0] = 8
        elif person[0] == "150,000 or more":
            person[0] = 9

        # Education Level Conversion
        if person[1] == "Less than high school":
            person[1] = 1
        elif person[1] == "High school incomplete":
            person[1] = 2
        elif person[1] == "High school graduate":
            person[1] = 3
        elif person[1] == "Some college, no degree":
            person[1] = 4
        elif person[1] == "Two-year associate degree from a college or university":
            person[1] = 5
        elif person[1] == "Four-year college or university degree/Bachelor's degree":
            person[1] = 6
        elif person[1] == "Some postgraduate or professional schooling, no postgraduate degree":
            person[1] = 7
        elif person[1] == "Postgraduate or professional schooling, including master's, doctorate, medical, or law degree":
            person[1] = 8

        # Parent Conversion
        if person[2] == "Yes" :
            person[2] = 1
        elif person[2] == "No":
            person[2] = 0

        # Married Conversion
        if person[3] == "Married" :
            person[3] = 1
        elif person[3] == "Not Married":
            person[3] = 0

        # Gender Conversion
        if person[4] == "Female" :
            person[4] = 1
        elif person[4] != "":
            person[4] = 0
        
        pred_class = model1.predict([person])
        pred_prob = model1.predict_proba([person])
        
        if pred_class == 0:
            st.write(f"Is {name} a LinkedIn user? **No.**")
        else:
            st.write(f"Is {name} a LinkedIn user? **YES!**")
        st.write(f"Probability that {name} is a LinkedIn user: {round(pred_prob[0][1]*100, 2)}%")

    except ValueError:
        st.error("Please enter all information.")

name = st.text_input("What is your name? (Optional)", value = "")    
user_income = st.selectbox("What is your current income level?", 
                           options=["",
                                    "Less than 10000",
                                    "10,000 to under 20,000",
                                    "20,000 to under 30,000",
                                    "30,000 to under 40,000",
                                    "40,000 to under 50,000",
                                    "50,000 to under 75,000",
                                    "75,000 to under 100,000",
                                    "100,000 to under 150,000",
                                    "150,000 or more"])
user_education = st.selectbox("What is your education level?", 
                              options=["",
                                       "Less than high school",
                                        "High school incomplete",
                                        "High school graduate",
                                        "Some college, no degree",
                                        "Two-year associate degree from a college or university",
                                        "Four-year college or university degree/Bachelor's degree",
                                        "Some postgraduate or professional schooling, no postgraduate degree",
                                        "Postgraduate or professional schooling, including master's, doctorate, medical, or law degree"
                                        ])
user_parent = st.radio("Are you a parent of a child under 18 living in your home?",
                        options=["Yes", "No"])
user_married = st.radio("What is your current marital status?:", 
                        options=["Married", "Not Married"])
user_female = st.radio("What is your gender?", 
                       options=["Male", "Female"])
user_age = st.slider("How old are you?", 
                     min_value=1, max_value=97, step=1)

if st.button("Submit"):
    sent_app(name, user_income, user_education, user_parent, user_married,
             user_female, user_age)