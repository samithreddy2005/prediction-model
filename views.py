# predictor/views.py
from django.shortcuts import render
import joblib
import numpy as np

def predict_price(request):
    if request.method == "POST":
        # Get user inputs from the form
        category = int(request.POST["category"])  # 1 for Electronics, 2 for Clothing
        brand = int(request.POST["brand"])  # 1 for Samsung, 2 for Nike, etc.
        rating = float(request.POST["rating"])

        # Load the trained model
        model = joblib.load("predictor/product_price_model.pkl")
        
        # Predict the price
        prediction = model.predict(np.array([[category, brand, rating]]))
        return render(request, "predict_price.html", {"prediction": prediction[0]})

    return render(request, "predict_price.html")
