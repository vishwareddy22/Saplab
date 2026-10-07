<<<<<<< HEAD
# Saplab
=======
# Customer Churn Prediction API
 
This project predicts whether a telecom customer is likely to leave the
company using a Random Forest machine learning model, served with FastAPI
and deployed on Render.
## Run Locally
pip install -r requirements.txt
python train_model.py
uvicorn app:app --reload
 
## API
- GET  /         health check
- POST /predict  returns a churn prediction for a JSON customer record
>>>>>>> d5f4595 (Adding files to repo)
