from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
 
app = FastAPI(title="Customer Churn Prediction API")
 
# Load the trained pipeline once at startup, not per request.
model = joblib.load("churn_model.joblib")
 
@app.get("/")
def home():
    """Simple health-check endpoint."""
    return {"message": "Customer Churn Prediction API Running"}
 
# Request schema: every feature the model was trained on.
class CustomerRecord(BaseModel):
    Gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    Tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
 
@app.post("/predict")
def predict(customer: CustomerRecord):
    # Convert the validated request body into a single-row DataFrame. Column names must match those seen in training, since the saved pipeline selects columns by name.
    row = pd.DataFrame([customer.dict()])
 
    # The pipeline re-applies the same one-hot encoding used during training, so text fields can be passed in as-is.
    prediction = model.predict(row)[0]
    probability = model.predict_proba(row)[0][1]
 
    result = "Customer Will Leave" if prediction == 1 else "Customer Will Stay"
 
    return {
        "Prediction": result,
        "churn_probability": round(float(probability), 2),
    }