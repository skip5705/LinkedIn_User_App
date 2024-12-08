#### Set-Up -------------------

# Import Packages
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

#### Test Predictions ---------

# Person 1 test: Age = 42
model1_predict(8, 7, 0, 1, 1, 42)

# Person 2 test: Age = 84
model1_predict(8, 7, 0, 1, 1, 84)