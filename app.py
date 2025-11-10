import streamlit as st
import pickle
import pandas as pd

@st.cache_resource
def load_model():
    with open('model1.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
st.title("Placement Prediction App")

def get_user_input():
    col_id = st.sidebar.text_input('College_ID', value='Unknown', key='college_id_input')
    cgpa = st.sidebar.number_input('CGPA', 0.0, 10.0, step=0.1, key='cgpa')
    iq = st.sidebar.number_input('IQ', 0, 200, step=1, key='iq')
    prev_sem_res = st.sidebar.number_input('Prev_Sem_Result', 0.0, 10.0, step=0.1, key='prev_sem')
    academic_perf = st.sidebar.number_input('Academic_Performance', 0.0, 10.0, step=0.1, key='academic_perf')
    extra_curricular_score = st.sidebar.number_input('Extra_Curricular_Score', 0.0, 10.0, step=0.1, key='extra_curr')
    communication_skills = st.sidebar.number_input('Communication_Skills', 0.0, 10.0, step=0.1, key='comm_skills')
    projects_completed = st.sidebar.number_input('Projects_Completed', 0, 7, step=1, key='projects')
    intern_experience = st.sidebar.selectbox('Internship_Experience', ['Yes', 'No'], key='intern')

    return pd.DataFrame([{
        'College_ID': col_id,
        'CGPA': cgpa,
        'IQ': iq,
        'Internship_Experience': intern_experience,
        'Prev_Sem_Result': prev_sem_res,
        'Academic_Performance': academic_perf,
        'Extra_Curricular_Score': extra_curricular_score,
        'Communication_Skills': communication_skills,
        'Projects_Completed': projects_completed
    }])

input_df = get_user_input()
st.subheader("Your Input")
st.write(input_df)

def default_for(col):
    if col in ['CGPA','IQ','Prev_Sem_Result','Academic_Performance',
               'Extra_Curricular_Score','Communication_Skills','Projects_Completed']:
        return 0.0
    if col in ['Internship_Experience','College_ID']:
        return 'Unknown'
    return ''

def prepare_input(df):
    expected = model.named_steps['preprocessor'].feature_names_in_
    for col in expected:
        if col not in df.columns:
            df[col] = default_for(col)
    return df[expected]

prepared = prepare_input(input_df)

@st.cache_data
def predict(df):
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)
    return pred, proba

if st.button("Predict"):
    pred_label, proba = predict(prepared)
    st.success(f"Placement: {pred_label}")
    st.write(pd.DataFrame(proba, columns=model.classes_))
