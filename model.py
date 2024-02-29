import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import pickle

# Load the dataset
df = pd.read_csv("data.csv")

# Perform one-hot encoding for categorical variables
df_encoded = pd.get_dummies(df, columns=['Brand', 'Model', 'Condition'])

# Save the encoded dataframe to a new CSV file
df_encoded.to_csv("df_encoded.csv", index=False)

# Split the dataset into features (X) and target variable (y)
X = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']

# Define the list of feature variables
var_vehicles = list(df_encoded.columns)
var_vehicles.remove('Price')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the RandomForestRegressor model
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Make predictions on the testing set
predictions = random_forest.predict(X_test)

# Calculate Mean Squared Error (MSE) and R-squared (R²) scores
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

# Save the trained RandomForestRegressor model using joblib
joblib.dump(random_forest, "random_forest_model.pkl")

# Alternative method to save the model using pickle
"""
pickle.dump(random_forest, open('model.pkl','wb'))
"""
