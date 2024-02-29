import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("data.csv")

df_encoded = pd.get_dummies(df,columns=['Brand','Model','Condition'])


df_encoded.to_csv("df_encoded.csv", index=False)

X = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']

var_vehicles = list(df_encoded.columns)
var_vehicles.remove('Price')

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

predictions = random_forest.predict(X_test)


mse = mean_squared_error(y_test,predictions)
r2 = r2_score(y_test,predictions)


print("Error cuadrático medio (MSE):", mse)
print("Coeficiente de determinación (R²):", r2)


joblib.dump(random_forest , "random_forest_model.pkl")
"""
pickle.dump(random_forest, open('model.pkl','wb'))
"""