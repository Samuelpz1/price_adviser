import pandas as pd
import joblib


usuario_datos = {
    'Brand': ' Toyota ',
    'Model': 'F-150',
    'Year': 2018,
    'Mileage': 35000,
    'Condition': 'Excellent'
}

usuario_df = pd.DataFrame(usuario_datos, index = [0])


usuario_df = pd.get_dummies(usuario_df,columns=['Brand','Model','Condition'])


df_encoded = pd.read_csv("df_encoded.csv")
df_encoded_columns = df_encoded.columns.tolist()
missing_columns = set(df_encoded.columns) - set(usuario_df.columns)
for column in missing_columns:
    usuario_df[column]= False

usuario_df = usuario_df[df_encoded_columns]

if 'Price' in usuario_df.columns:
    usuario_df = usuario_df.drop('Price', axis=1)



model = joblib.load("random_forest_model.pkl")

predicted_price = model.predict(usuario_df)
print(predicted_price)