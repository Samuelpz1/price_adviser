# Importing necessary libraries
import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image
import joblib
import pandas as pd

# Load the machine learning model
model = joblib.load("random_forest_model.pkl")

# Function to load Lottie animations from a URL
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load a Lottie animation from a URL and display it
lottie_coding = load_lottie_url("https://lottie.host/4d84de5d-2c3f-4893-ba22-02724a91a1cb/BKbeXCtsTG.json")
st_lottie(lottie_coding, height=300, key="car")

# Sidebar with input widgets for user interaction
st.sidebar.header("PRICE ADVISOR")
brand = st.sidebar.selectbox("Vehicle's brand", ['Chevrolet', 'Ford', 'Honda', 'Nissan', 'Toyota'], 1)
selected_model = st.sidebar.selectbox("Vehicle's model", ['F-150', 'Civic', 'Altima', 'Camry', 'Silverado'], 1)
year = st.sidebar.slider("Vehicle's Year", 2010, 2024, 2017)
mileage = st.sidebar.slider("Vehicle's mileage", 0, 160000, 35000)
selected_condition = st.sidebar.selectbox("Vehicle's condition", ['Excellent', 'Good', 'Fair'], 1)

# Create a DataFrame with selected inputs
input_data = pd.DataFrame({
    'Brand': [brand],
    'Model': [selected_model],
    'Year': [year],
    'Mileage': [mileage],
    'Condition': [selected_condition]
})

# Encode categorical variables
input_data = pd.get_dummies(input_data, columns=['Brand', 'Model', 'Condition'])

# Ensure input data columns match the training data columns
df_encoded = pd.read_csv('df_encoded.csv')
df_encoded_columns = df_encoded.columns.tolist()
missing_columns = set(df_encoded.columns) - set(input_data.columns)
for column in missing_columns:
    input_data[column] = False

missing_columns = set(df_encoded.columns) - set(input_data.columns)

# Reorder columns to match the training data
input_data = input_data[df_encoded_columns]

# Remove 'Price' column from input data
if 'Price' in input_data.columns:
    input_data = input_data.drop('Price', axis=1)

# Make prediction using the model
predicted_price = model.predict(input_data)

# Display the predicted price range to the user
st.markdown("<h1 style= 'text-align: center; ' > Price Advisor</h1>", unsafe_allow_html=True)
min_predicted = predicted_price * 0.9
max_predicted = predicted_price * 1.1
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown("")
        st.markdown("")
        st.write("According to data provided you could sell your car between:")
    with right_column:
        r_left_column, r_right_column = st.columns(2)
        with r_left_column:
            st.header("Min Price")
            st.write(f"${'{:,.2f}'.format(min_predicted[0])}")
        with r_right_column:
            st.header("Max Price")
            st.write(f"${'{:,.2f}'.format(max_predicted[0])}")