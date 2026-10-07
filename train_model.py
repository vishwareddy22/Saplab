import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
 
# 1. Load the dataset
data = pd.read_excel("customer_churn.xlsx")
 
# 2. Separate features (X) and target (y)
# Use every column except the target, instead of a small hand-picked subset, so the model can use all available signals.
X = data.drop(columns=["Churn"])
 
# The target is text ("Yes"/"No"); map it to 1/0 for the classifier and so the API's output is unambiguous.
y = data["Churn"].map({"Yes": 1, "No": 0})
 
# 3. Identify categorical vs. numerical columns
categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()
 
# 4. Preprocessing: one-hot encode categorical columns
# Contract, InternetService, PaymentMethod and the other text columns cannot be passed directly into RandomForestClassifier, as the original script attempted. handle_unknown="ignore" also protects the live API from crashing on a category value that was not present during training.
preprocessor = ColumnTransformer(transformers=[
    ("num", "passthrough", numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])
 
# 5. Combine preprocessing and model into a single pipeline
# Saving preprocessing and the classifier together means app.py does not need to re-implement any encoding logic.
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
])
 
# 6. Split into training and test sets, then train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
 
# 7. Evaluate and save
accuracy = model.score(X_test, y_test)
print(f"Model Trained Successfully. Test Accuracy: {accuracy:.2f}")
 
joblib.dump(model, "churn_model.joblib")
print("Model Saved Successfully")