# predictor/product_price_predictor.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample dataset
data = {
    "category": ["Electronics", "Clothing", "Electronics", "Clothing"],
    "brand": ["Samsung", "Nike", "Apple", "Adidas"],
    "rating": [4.5, 4.0, 4.7, 4.3],
    "price": [500, 30, 700, 40]
}

# Convert the categorical data into numerical values
df = pd.DataFrame(data)
df["category"] = df["category"].map({"Electronics": 1, "Clothing": 2})
df["brand"] = df["brand"].map({"Samsung": 1, "Nike": 2, "Apple": 3, "Adidas": 4})

# Features and target
X = df[["category", "brand", "rating"]]
y = df["price"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model for later use
joblib.dump(model, "product_price_model.pkl")
