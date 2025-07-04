import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import os

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="🏦", # Bank icon
    layout="centered", # Can also be "wide"
    initial_sidebar_state="auto" # Can also be "expanded" or "collapsed"
)

# --- Load the Model and Encoders ---
# Get the directory of the current script
script_dir = os.path.dirname(__file__)

try:
    model_path = os.path.join(script_dir, 'model.h5')
    model = tf.keras.models.load_model(model_path)

    onehot_encoder_geo_path = os.path.join(script_dir, 'onehot_encoder_geop.pkl')
    label_encoder_gender_path = os.path.join(script_dir, 'label_encoder_gender.pkl')
    scaler_path = os.path.join(script_dir, 'scaler.pkl')

    with open(onehot_encoder_geo_path, 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    with open(label_encoder_gender_path, 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

except FileNotFoundError as e:
    st.error(f"Error loading required files: {e}. Please ensure 'model.h5', 'onehot_encoder_geop.pkl', 'label_encoder_gender.pkl', and 'scaler.pkl' are in the same directory as 'app.py'.")
    st.stop() # Stop the app if essential files are missing
except Exception as e:
    st.error(f"An unexpected error occurred while loading files: {e}")
    st.stop()

# --- Streamlit App UI ---

st.markdown(
    """
    <style>
    .main {
        background-color: #F0F2F6; /* Light gray background */
        padding: 20px;
        border-radius: 10px;
    }
    .stSelectbox, .stSlider, .stNumberInput {
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    h1 {
        color: #4CAF50; /* Green for main title */
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 20px;
    }
    h2 {
        color: #336699; /* Blue for section titles */
        font-size: 1.8em;
        border-bottom: 2px solid #336699;
        padding-bottom: 5px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .stButton > button {
        background-color: #4CAF50; /* Green button */
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stAlert {
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('🏦 Customer Churn Prediction 📉')
st.markdown("---") # Horizontal line for separation

st.subheader('Enter Customer Details Below:')

# --- User Input Layout using Columns ---

# Row 1: Geography, Gender, Age
col1, col2, col3 = st.columns(3)
with col1:
    geography = st.selectbox('🌎 Geography', onehot_encoder_geo.categories_[0])
with col2:
    gender = st.selectbox('🚻 Gender', label_encoder_gender.classes_)
with col3:
    age = st.slider('📅 Age', 18, 92, 35) # Added default value

# Row 2: Credit Score, Balance, Estimated Salary
col4, col5, col6 = st.columns(3)
with col4:
    credit_score = st.number_input('🌟 Credit Score', min_value=300, max_value=850, value=650)
with col5:
    balance = st.number_input('💰 Balance', min_value=0.0, value=50000.0)
with col6:
    estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0, value=60000.0)

# Row 3: Tenure, Number of Products, Has Credit Card, Is Active Member
col7, col8 = st.columns(2)
with col7:
    tenure = st.slider('🗓️ Tenure (Years)', 0, 10, 5)
    num_of_products = st.slider('🛍️ Number of Products', 1, 4, 1)
with col8:
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.selectbox('🚶 Active Member', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.markdown("---") # Another horizontal line

# --- Prediction Button ---
if st.button('Predict Churn'):
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    # Suppress the UserWarning related to feature names for OneHotEncoder
    # by making sure the input is a DataFrame with column names if possible
    # or by wrapping the transform with a context manager if you absolutely
    # want to suppress it, but it's often better to understand it.
    # For now, we'll keep it as is, as it's just a warning.

    geo_encoded = onehot_encoder_geo.transform(pd.DataFrame([[geography]], columns=['Geography'])).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data
    # Ensure correct column order for scaling (match training data)
    # This assumes the original training data columns were in a specific order.
    # A robust solution would be to save the column order from training.
    # For now, let's assume the original order was:
    # CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography_France, Geography_Germany, Geography_Spain

    # Create a DataFrame with all expected columns and fill them
    # First, get all original feature names including one-hot encoded ones
    all_features = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'] + list(onehot_encoder_geo.get_feature_names_out(['Geography']))
    processed_input_df = pd.DataFrame(0, index=[0], columns=all_features) # Initialize with zeros

    # Fill numerical and label-encoded features
    for col in input_data.columns:
        processed_input_df[col] = input_data[col]

    # Fill one-hot encoded geography features
    for col in geo_encoded_df.columns:
        processed_input_df[col] = geo_encoded_df[col]

    # Scale the input data
    input_data_scaled = scaler.transform(processed_input_df)

    # Predict churn
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.subheader('Prediction Result:')
    st.write(f'**Churn Probability: {prediction_proba:.2f}**')

    if prediction_proba > 0.5:
        st.error('The customer is **likely to churn**! 🚨')
    else:
        st.success('The customer is **not likely to churn**. ✅')

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and TensorFlow.")