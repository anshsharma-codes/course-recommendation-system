# Import Essential Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# Load the dataset
df=pd.read_excel("MIS Data.xlsx")
df.columns=df.columns.str.strip().str.replace(" ","_").str.replace("-","_")
df['College']=df['College'].fillna('not provided')
df['Branch']=df['Branch'].str.strip().str.title()
df['Course']=df['Course'].str.strip().str.title()
df['Subject']=df['Subject'].str.strip().str.title()

# Encode features and train MLmodel for preduction
features=['Branch','College','Course','Year']
target='Subject'
df_ml=df.dropna(subset=features+[target])
encoders={}
for col in features+[target]:
    le=LabelEncoder()
    df_ml[col]=le.fit_transform(df_ml[col])
    encoders[col]=le
x=df_ml[features]
y=df_ml[target]
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)

#preduction function using Ml Model
def predict_subjects_ml(branch,college,course,year,top_n=3):
        input_df=pd.DataFrame([[branch,college,course,year]],columns=features)
        for col in features:
              input_df[col]=encoders[col].transform(input_df[col])
        probs=model.predict_proba(input_df)[0]
        top_indices=np.argsort(probs)[::-1][:top_n]
        subject_names=encoders[target].inverse_transform(top_indices)
        return list(zip(subject_names,probs[top_indices]))



# UI Designing
st.title("Course Recomendetion system")
st.markdown("Get top recomended course based on your branch,collage,course and Year")

branches=sorted(df['Branch'].dropna().unique())
colleges=sorted(df['College'].dropna().unique())
courses=sorted(df['Course'].dropna().unique())
years=sorted(df['Year'].dropna().unique())


selcted_branch=st.selectbox("Select Branch",branches)
selcted_colleg=st.selectbox("Select College",colleges)
selcted_course=st.selectbox("Select Course",courses)
selcted_year=st.selectbox("Select Year",years)
btn=st.button("Recommend Course(ML-Based)")
if btn:
    ml_recommendations=predict_subjects_ml(selcted_branch,selcted_colleg,selcted_course,selcted_year)
    st.subheader("ML Recomended subjects")
    for i,(subjects,score) in enumerate(ml_recommendations,1):
          st.markdown(f"{i}.**{subjects}** -- Confidence:{score:.2f}")