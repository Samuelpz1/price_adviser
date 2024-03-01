# Importing necessary libraries
import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image
import joblib
import pandas as pd

# Load the machine learning model
modelo = joblib.load("random_forest_model.pkl")

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

df = pd.read_csv("data.csv")
makes = df['Make'].unique()
states = df['State'].unique()
model_dict = {}
for maker in makes:
    model_dict[maker] = []

for index, row in df[['Make','Model']].iterrows():
    make = row['Make']
    model = row['Model']
    if model not in model_dict[make]:
        model_dict[make].append(model)

state = st.sidebar.selectbox("State", states, 1)
brand = st.sidebar.selectbox("Vehicle's manufacturer", makes, 1)
selected_model = st.sidebar.selectbox("Vehicle's model", model_dict[brand], 1)
year = st.sidebar.slider("Vehicle's Year", 2010, 2024, 2017)
mileage = st.sidebar.number_input("Vehicle's mileage", 0, None)

#sidebar.slider("Vehicle's mileage", 0, df['Mileage'].max(), 35000)


# Create a DataFrame with selected inputs
input_data = pd.DataFrame({
    'Brand': [brand],
    'Model': [selected_model],
    'Year': [year],
    'Mileage': [mileage],
    'State': [state]
})

# Encode categorical variables
input_data = pd.get_dummies(input_data, columns=['Brand', 'Model', 'State'])

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
predicted_price = modelo.predict(input_data)

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