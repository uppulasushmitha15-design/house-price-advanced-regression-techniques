# house-price-advanced-regression-techniques
Data Science project using Python and Machine Learnling
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("train.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)
print("\nFirst 5 Rows:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nStatistical Summary:")
summary = df.describe()
print(summary)
print("\nCorrelation with SalePrice:")
corr = df.corr(numeric_only=True)["SalePrice"]
corr = corr.sort_values(ascending=False)

print(corr)
print("\nTop 10 features Correlated with SalePrice:")
print(corr.head(10))
top_features = corr.head(10).index
plt.figure(figsize=(10,8))
sns.heatmap(df[top_features].corr(),
            annot=True,
            cmap="coolwarm")
features = features = [
    'OverallQual',
    'GrLivArea',
    'GarageCars',
    'GarageArea',
    'TotalBsmtSF',
    '1stFlrSF',
    'FullBath',
    'TotRmsAbvGrd',
    'YearBuilt'
]

X = df[features]
y = df['SalePrice']
X = df[top_features]
Y = df["SalePrice"]
X = X.fillna(X.mean())
print(df.isnull().sum())
print("Total Missing Values:",
df.isnull().sum().sum())
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(y_pred[:10])
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))
comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print(comparison.head(10))
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
plt.hist(df["SalePrice"] , bins = 30)
plt.title("House Price Distribution")
plt.xlabel("Sale Price")
plt.ylabel("count")
plt.show()


