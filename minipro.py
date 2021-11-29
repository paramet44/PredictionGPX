import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.write("""
# Student Questionnaire
 **Forecast grades from emotional health**
""")


st.write("""**1. Relationship Level**""")
st.write("""Level of relationship with close people""")
V_Relation_fam = st.slider('1.1 Family', min_value=-5, max_value=5, value=0)
V_Relation_friend = st.slider(
    '1.2 Friends at university', min_value=-5, max_value=5, value=0)
V_Relation_aj = st.slider('1.3 Teacher at university',
                          min_value=-5, max_value=5, value=0)


st.write("""**2. The level of feelings towards the university environment**""")
st.write("""The level of feelings towards the various service departments of the university""")
V_Feel_uni_1 = st.slider(
    '2.1 Feelings towards the teaching and learning management of the school', min_value=-5, max_value=5, value=0)
V_Feel_uni_2 = st.slider(
    '2.2 Feelings towards student development ', min_value=-5, max_value=5, value=0)
V_Feel_uni_3 = st.slider(
    '2.3 Feelings towards the Student Services of the Registrar', min_value=-5, max_value=5, value=0)
V_Feel_uni_4 = st.slider(
    '2.4 Feelings for Student Services of the Finance Department', min_value=-5, max_value=5, value=0)
V_Feel_uni_5 = st.slider(
    '2.5 Feelings towards Student Services of the Library Center (Library)', min_value=-5, max_value=5, value=0)
V_Feel_uni_6 = st.slider(
    '2.6 Feelings towards the Student Services of the Information Technology Service Center', min_value=-5, max_value=5, value=0) 
V_Feel_envi_7 = st.slider(
    '2.7 The feeling of the student service of the university dormitory', min_value=-5, max_value=5, value=0)
V_Feel_envi_8 = st.slider(
    '2.8 Feelings towards Student Services of Mae Fah Luang University Hospital', min_value=-5, max_value=5, value=0)
V_Feel_envi_9 = st.slider(
    '2.9 Feelings towards the Student Services of the Counseling Office', min_value=-5, max_value=5, value=0)
V_Feel_envi_10 = st.slider(
    '2.10 The feeling of student service at the cafeteria', min_value=-5, max_value=5, value=0)
V_Feel_envi_11 = st.slider(
    '2.11 The feeling of student service of the transportation system within the university.', min_value=-5, max_value=5, value=0)
V_Feel_envi_12 = st.slider(
    '2.12 Feelings towards the university sports center', min_value=-5, max_value=5, value=0)
V_Feel_envi_13 = st.slider(
    '2.13 The feeling of the buildings in the university (Reading books, relaxing, doing activities)', min_value=-5, max_value=5, value=0)
V_Feel_envi_14 = st.slider(
    '2.14 Environmental feeling on university', min_value=-5, max_value=5, value=0)


st.write("""**3. Problems students faced during the last semester**""")
V_Problem = st.radio('Is there a problem?', ['Yes', 'No'])
V_Problem_study = st.radio('3.1 Learning', ['Yes', 'No'])
V_Problem_love = st.radio('3.2 Love', ['Yes', 'No'])
V_Problem_adjust = st.radio('3.3 Adaptation', ['Yes', 'No'])
V_Problem_econ = st.radio('3.4 Economy', ['Yes', 'No'])
V_Problem_other = st.radio('3.5 Other', ['Yes','No'])
st.text_input('Input: ')
st.write("""**4. Attitudes towards school/major that studing**""")
V_Attitude = st.radio('Input:', ['Good', 'Default', 'Bad'])

# Change the value of Problem to be {'0','1'} as stored in the trained dataset
if V_Problem == 'Yes':
    V_Problem = 1
else:
    V_Problem = 0

# Change the value of Problem to be {'0','1'} as stored in the trained dataset
if V_Problem_study == 'Yes':
    V_Problem_study = 1
else:
    V_Problem_study = 0

# Change the value of Problem to be {'0','1'} as stored in the trained dataset
if V_Problem_love == 'Yes':
    V_Problem_love = 1
else:
    V_Problem_love = 0

# Change the value of Problem to be {'0','1'} as stored in the trained dataset
if V_Problem_adjust == 'Yes':
    V_Problem_adjust = 1
else:
    V_Problem_adjust = 0

# Change the value of Problem to be {'0','1'} as stored in the trained dataset
if V_Problem_econ == 'Yes':
    V_Problem_econ = 1
else:
    V_Problem_econ = 0

# Change the value of Problem to be {'0','1'} as stored in the trained dataset
if V_Problem_other == 'Yes':
    V_Problem_other = 1
else:
    V_Problem_other = 0

# Change the value of Attitude to be {'0','1','2'} as stored in the trained dataset
if V_Attitude == 'Good':
    V_Attitude = 2
elif V_Attitude == 'Default':
    V_Attitude = 1
else:
    V_Attitude = 0

data = {
    'Relation_fam': V_Relation_fam,
    'Relation_friend': V_Relation_friend,
    'Relation_aj': V_Relation_aj,
    'Feel_uni_1': V_Feel_uni_1,
    'Feel_uni_2': V_Feel_uni_2,
    'Feel_uni_3': V_Feel_uni_3,
    'Feel_uni_4': V_Feel_uni_4,
    'Feel_uni_5': V_Feel_uni_5,
    'Feel_uni_6': V_Feel_uni_6,
    'Feel_envi_7': V_Feel_envi_7,
    'Feel_envi_8': V_Feel_envi_8,
    'Feel_envi_9': V_Feel_envi_9,
    'Feel_envi_10': V_Feel_envi_10,
    'Feel_envi_11': V_Feel_envi_11,
    'Feel_envi_12': V_Feel_envi_12,
    'Feel_envi_13': V_Feel_envi_13,
    'Feel_envi_14': V_Feel_envi_14,
    'Problem': V_Problem,
    'Problem_study': V_Problem_study,
    'Problem_adjust':V_Problem_adjust,
    'Problem_love': V_Problem_love,
    'Problem_econ': V_Problem_econ,
    'Problem_other':V_Problem_other,
    'Attitude': V_Attitude
}
df = pd.DataFrame(data, index=[0])

# combine all data
X_new = pd.concat([df], axis=1)

# Select only the first row (the user input data)
X_new = X_new[:1]

# --Reads the saved normalization model
load_nor = pickle.load(open('normalization.pkl', 'rb'))
X_new = load_nor.transform(X_new)

# -- Reads the saved classification model
load_knn = pickle.load(open('LinearRegression.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X_new)
# Show the prediction result on the screen
st.subheader('GPX Prediction :')
st.write(prediction)
