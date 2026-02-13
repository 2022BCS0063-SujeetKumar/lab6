import json
import joblib
import os

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Create required directories
os.makedirs("Lab/model", exist_ok=True)
os.makedirs("Lab/app/artifacts", exist_ok=True)

# Load dataset
X, y = load_wine(return_X_y=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Save model
joblib.dump(model, "Lab/model/model.pkl")

# Save metrics
metrics = {
    "r2": float(r2),
    "mse": float(mse)
}

with open("Lab/app/artifacts/metrics.json", "w") as f:
    json.dump(metrics, f)

print("Training complete")
print("R2 Score:", r2)
print("MSE:", mse)
