import streamlit as st
from prediction import *

raw_data = pd.DataFrame()

st.title(":school: Jaya Jaya Institute Dropout Prediction :school:")

# Personal Information section
st.header("Personal Information", divider="rainbow")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(label="Gender", options=["Laki-laki","Perempuan"], index=0)
    raw_data["Gender"] = [gender]
    
with col2:
    age_at_enrollment = st.number_input("Age at Enrollment", value=22)
    raw_data["Age_at_enrollment"] = [age_at_enrollment]
    
    
col1, col2, col3 = st.columns(3)
    
with col1:
    debtor = st.selectbox(label="Debtor", options=["Tidak","Ya"], index=0)
    raw_data["Debtor"] = [debtor]
    
with col2:
    scholarship_holder = st.selectbox(label="Scholarship Holder", options=["Tidak","Ya"], index=0)
    raw_data["Scholarship_holder"] = [scholarship_holder]
with col3:
    tuition_fees = st.selectbox(label="Tuition Fees Up To Date", options=["Tidak","Ya"], index=0)
    raw_data["Tuition_fees_up_to_date"] = [tuition_fees]

# Curricular Units section
## 1st Sem section
st.header("Curricular Units 1st Semester Information", divider="rainbow")

col1, col2, col3, col4 = st.columns(4)

with col1:
    curr_1st_enrolled = int(st.number_input(label="Enrolled (0 -30)", min_value=0, max_value=30, value=20, key="1st sem enrolled"))
    raw_data["Curricular_units_1st_sem_enrolled"] = [curr_1st_enrolled]

with col2:
    curr_1st_eval = int(st.number_input(label="Evaluations (0 - 50)", min_value=0, max_value=50, value=35, key="1st sem evaluations"))
    raw_data["Curricular_units_1st_sem_evaluations"] = [curr_1st_eval]
    
with col3:
    curr_1st_approved = int(st.number_input(label="Approved (0 - 30)", min_value=0, max_value=30, value=20, key="1st sem approved"))
    raw_data["Curricular_units_1st_sem_approved"] = [curr_1st_approved]
    
with col4:
    curr_1st_grade = float(st.number_input(label="Grade (0 - 20)", min_value=0.0, max_value=20.0, value=17.0, key="1st sem grade"))
    raw_data["Curricular_units_1st_sem_grade"] = [curr_1st_grade]
    
## 2nd Semester section
st.header("Curricular Units 2nd Semester Information", divider="rainbow")

col1, col2, col3, col4 = st.columns(4)

with col1:
    curr_2nd_enrolled = int(st.number_input(label="Enrolled (0 -30)", min_value=0, max_value=30, value=20, key="2nd sem enrolled"))
    raw_data["Curricular_units_2nd_sem_enrolled"] = [curr_2nd_enrolled]
    
with col2:
    curr_2nd_eval = int(st.number_input(label="Evaluations (0 - 50)", min_value=0, max_value=50, value=35, key="2nd sem evaluations"))
    raw_data["Curricular_units_2nd_sem_evaluations"] = [curr_2nd_eval]
    
with col3:
    curr_2nd_approved = int(st.number_input(label="Approved (0 - 30)", min_value=0, max_value=30, value=20, key="2nd sem approved"))
    raw_data["Curricular_units_2nd_sem_approved"] = [curr_2nd_approved]
    
with col4:
    curr_2nd_grade = float(st.number_input(label="Grade (0 - 20)", min_value=0.0, max_value=20.0, value=17.0, key="2nd sem grade"))
    raw_data["Curricular_units_2nd_sem_grade"] = [curr_2nd_grade]

with st.expander("Overall Information"):
    st.dataframe(data=raw_data, width=800, height=10)


data_input = np.array([
    curr_1st_enrolled,
    curr_2nd_enrolled,
    curr_1st_approved,
    curr_2nd_approved,
    curr_1st_grade,
    curr_2nd_grade,
    curr_1st_eval,
    curr_2nd_eval,
    age_at_enrollment,
    debtor,
    tuition_fees,
    gender,
    scholarship_holder
])
if st.button("Predict"):
    data_preprocessed = data_preprocessing(data_input)
    result = prediction(data_preprocessed)
    if result == "Graduate":
        caption = "Congratsss!!! You are Graduated"
    elif result == "Dropout":
        caption = "Unfortunately, you are Dropouted"
    else:
        caption = "You are still enrolled"
    with st.expander("View the preprocessed data"):
        st.dataframe(data=data_preprocessed, width=800, height=10)
    st.write(caption)
