from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model/model.pkl")

@app.get("/")
def root():
    return {"message": "Lab 6 Model API is running"}

@app.post("/predict")
def predict(features: list):
    data = np.array(features).reshape(1, -1)
    prediction = int(model.predict(data)[0])

    return {
        "name": "Sujeet Kumar",
        "roll_no": "2022BCS0063",
        "wine_quality": prediction
    }
