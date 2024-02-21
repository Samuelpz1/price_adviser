import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image
import joblib
import pandas as pd

model = joblib.load("random_forest_model.pkl")

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottie_url("https://lottie.host/4d84de5d-2c3f-4893-ba22-02724a91a1cb/BKbeXCtsTG.json")
st_lottie(lottie_coding,height = 300 , key = "car")

st.sidebar.header("PRICE ADVISOR")

brand = st.sidebar.selectbox("Vechicle's brand",['Chevrolet','Ford','Honda','Nissan','Toyota'],1)
selected_model = st.sidebar.selectbox("Vechicle's model",['F-150','Civic','Altima','Camry','Silverado'],1)
year = st.sidebar.slider("Vehicle's Year",2010 ,2024, 2017)
mileage = st.sidebar.slider("Vechicle's mileage",0,160000,35000)
selected_condition = st.sidebar.selectbox("Vechicle's condition",['Excellent','Good','Fair'],1)

input_data = pd.DataFrame( {
    'Brand' : [brand],
    'Model' : [selected_model],
    'Year' : [year],
    'Mileage' : [mileage],
    'Condition' : [selected_condition]
    })

#codificar las variales categoricas
input_data = pd.get_dummies(input_data, columns=['Brand','Model','Condition'])

# Asegurarse de que las columnas coincidan con las columnas utilizadas durante el entrenamiento
df_encoded = pd.read_csv('df_encoded.csv')
df_encoded_columns = df_encoded.columns.tolist()
missing_columns = set(df_encoded.columns) - set(input_data.columns)
for column in missing_columns:
    input_data[column] = False

missing_columns = set(df_encoded.columns) - set(input_data.columns)

# Reordenar las columnas para que coincidan con el orden utilizado durante el entrenamiento
input_data = input_data[df_encoded_columns]

# Eliminar la columna 'Price' de los datos de entrada
if 'Price' in input_data.columns:
    input_data = input_data.drop('Price', axis=1)

print(input_data)
print(df_encoded_columns)
# Realizar la predicción utilizando el modelo
predicted_price = model.predict(input_data)
print(type(predicted_price[0]))

# Mostrar el resultado al usuario

#st.write(f"The predicted price of the vehicle is: ${'{:,.2f}'.format(predicted_price[0])}")

# st.title("Price Advisor")

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


