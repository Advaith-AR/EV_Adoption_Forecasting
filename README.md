import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Data Loading and Inspection
try:
    df = pd.read_csv("Electric_Vehicle_Population_Size_2024.csv")  # Replace with your actual path
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: The dataset file was not found. Double-check the path.")
    exit()

print("First 5 rows:\n", df.head())
print("\nDataset info:\n", df.info())
print("\nDescriptive statistics:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())

# 2. Data Cleaning and Preprocessing
df['Date'] = pd.to_datetime(df['Date'])
# Example: Fill missing numerical values with the mean
# df['Some_Numerical_Column'].fillna(df['Some_Numerical_Column'].mean(), inplace=True)
# Example: Drop rows with any missing values (use with caution)
# df.dropna(inplace=True)
print("\nMissing values after handling:\n", df.isnull().sum())

for col in ['County', 'State', 'Vehicle Primary Use']:
    print(f"\nUnique values in {col}:\n", df[col].unique())

df['Vehicle Primary Use'] = df['Vehicle Primary Use'].replace('Trucks', 'Truck')
df['Vehicle Primary Use'] = df['Vehicle Primary Use'].replace('Passenger', 'Passenger Vehicles')

print(f"\nUnique values in {col}:\n", df['Vehicle Primary Use'].unique())

# 3. Feature Engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Month_Year'] = df['Date'].dt.to_period('M')
df['Year_Month'] = df['Year'] + df['Month'] / 12

label_encoder = LabelEncoder()
df['County'] = label_encoder.fit_transform(df['County'])
df['State'] = label_encoder.fit_transform(df['State'])
df['Vehicle Primary Use'] = label_encoder.fit_transform(df['Vehicle Primary Use'])

print("\nFeature Engineered Data:\n", df.head())

# 4. Model Selection and Preparation
features = ['County', 'State', 'Vehicle Primary Use', 'Year', 'Month', 'Day', 'Year_Month']  # Add more relevant features
target = 'Electric Vehicle (EV) Total'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining set size:", len(X_train))
print("Testing set size:", len(X_test))

# 5. Model Training and Evaluation
rf_model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='neg_mean_squared_error'
)

random_search.fit(X_train, y_train)
best_rf_model = random_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")
print("\nBest Hyperparameters:", random_search.best_params_)

# 6. Saving the Model
joblib.dump(best_rf_model, 'ev_adoption_forecast_model.pkl')
print("\nModel saved as ev_adoption_forecast_model.pkl")

# 7. Forecasting Future EV Adoption
loaded_model = joblib.load('ev_adoption_forecast_model.pkl')
future_dates = pd.to_datetime(['2024-03-31', '2024-04-30', '2024-05-31'])
future_df = pd.DataFrame({'Date': future_dates})

future_df['Year'] = future_df['Date'].dt.year
future_df['Month'] = future_df['Date'].dt.month
future_df['Day'] = future_df['Date'].dt.day
future_df['Month_Year'] = future_df['Date'].dt.to_period('M')
future_df['Year_Month'] = future_df['Year'] + future_df['Month'] / 12

#THIS IS CRITICAL TO CHANGE!  See previous explanations for why this is a problem.
future_df['County'] = df['County'].mean()
future_df['State'] = df['State'].mean()
future_df['Vehicle Primary Use'] = df['Vehicle Primary Use'].mean()

future_X = future_df[features]
future_predictions = loaded_model.predict(future_X)

print("\nFuture Predictions:")
for date, prediction in zip(future_dates, future_predictions):
    print(f"{date.strftime('%Y-%m-%d')}: {prediction:.0f}")

# 8. Visualization
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Electric Vehicle (EV) Total'], label='Actual')
plt.plot(future_dates, future_predictions, label='Forecast', color='red')
plt.xlabel('Date')
plt.ylabel('Electric Vehicle (EV) Total')
plt.title('EV Adoption Forecast')
plt.legend()
plt.grid(True)
plt.show()