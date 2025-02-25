from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import random
import uvicorn

app = FastAPI()

# Load Models and Vectorizer
vectorizer = joblib.load("models/vectorizer.pkl")
baseline_model = joblib.load("models/baseline_model.pkl")
new_model = joblib.load("models/new_model.pkl")

class TextData(BaseModel):
    text: str

@app.post("/predict/")
def predict_sentiment(data: TextData):
    text_vector = vectorizer.transform([data.text])

    # A/B Testing Logic (50-50 split)
    if random.random() < 0.5:
        prediction = baseline_model.predict(text_vector)
        version = "Baseline Model (Naive Bayes)"
    else:
        prediction = new_model.predict(text_vector)
        version = "New Model (Logistic Regression)"

    return {
        "sentiment": prediction[0],
        "model_version": version
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
