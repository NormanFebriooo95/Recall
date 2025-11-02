import pandas as pd
import numpy as np
from joblib import load

cat_features_dict = {
    'Debtor': ['Tidak', 'Ya'],
    'Tuition_fees_up_to_date': ['Ya', 'Tidak'],
    'Gender': ['Perempuan', 'Laki-laki'],
    'Scholarship_holder': ['Ya', 'Tidak']
}

helper_df = pd.DataFrame(cat_features_dict)

# PCA
pca1 = load("assets/enroll")
pca2 = load("assets/approved")
pca3 = load("assets/grade")
pca4 = load("assets/evaluations")

# Transformer
transform_age = load("assets/Transformed_Age_at_enrollment")
transform_pca1 = load("assets/Transformed_pca1")
transform_pca2 = load("assets/Transformed_pca2")
transform_pca3 = load("assets/Transformed_pca3")
transform_pca4 = load("assets/Transformed_pca4")

# Model
tree_model = load("assets/tree_model.joblib")

transformers = [
    transform_age,
    transform_pca1,
    transform_pca2,
    transform_pca3,
    transform_pca4
]

def data_preprocessing(data_input, df=helper_df):
    # Split between the numeric data and categoric data
    numeric_data = data_input[:9]
    categoric_data = data_input[9:]
    
    # Numerical features preprocessing
    ## PCA
    pca1_result = list(pca1.transform([numeric_data[:2]])[0])  # Enrolled
    pca2_result = list(pca2.transform([numeric_data[2:4]])[0]) # Approved
    pca3_result = list(pca3.transform([numeric_data[4:6]])[0]) # Grade
    pca4_result = list(pca4.transform([numeric_data[6:8]])[0]) # Evaluations

    ## Power Transformer
    val_to_transformed_list = [*numeric_data[8], *pca1_result, *pca2_result, *pca3_result, *pca4_result]
    transformed_vals = []
    for transformer, val in zip(transformers, val_to_transformed_list):
        transformed_val = transformer.transform([[val]])[0][0]
        transformed_vals.append(transformed_val)
    
    # Categorical features preprocessing
    ## Add a new data to the dataframe
    df.loc[len(df)] = categoric_data

    ## Do a One-hot Encoding technique
    new_df = pd.get_dummies(df, dtype="int")

    ## Take the last index of the dataframe
    encoded_data_list = list(new_df.iloc[-1])

    # Concate both numeric processed data and categoric processed data
    # and save it into nummpy array
    preprocessed_data = pd.DataFrame([[*transformed_vals, *encoded_data_list]], columns=['Transformed_Age_at_enrollment',
                                                                                       'Transformed_pca1',
                                                                                       'Transformed_pca2',
                                                                                       'Transformed_pca3',
                                                                                       'Transformed_pca4',
                                                                                       'Debtor_Tidak',
                                                                                       'Debtor_Ya',
                                                                                       'Tuition_fees_up_to_date_Tidak',
                                                                                       'Tuition_fees_up_to_date_Ya',
                                                                                       'Gender_Laki-laki', 
                                                                                       'Gender_Perempuan', 
                                                                                       'Scholarship_holder_Tidak',
                                                                                       'Scholarship_holder_Ya'])
    
    return preprocessed_data

def prediction(preprocessed_data, model=tree_model):
    array = np.array(preprocessed_data)
    result = model.predict(array)[0]
    return result
