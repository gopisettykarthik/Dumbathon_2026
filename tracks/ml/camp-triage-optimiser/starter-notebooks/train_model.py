import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
from xgboost import plot_importance
import matplotlib.pyplot as plt



# STEP 1 — Load dataset
data = pd.read_csv("dataset.csv")

# STEP 2 — Separate features and target
X = data.drop("CaloriesNeeded", axis=1)
y = data["CaloriesNeeded"]

# STEP 3 — Convert text columns into numbers
X = pd.get_dummies(X)
feature_columns = X.columns

# STEP 4 — Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# STEP 5 — Build the model
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8
)

# STEP 6 — Train model
model.fit(X_train, y_train)

print("Model training completed")

# STEP 7 — Evaluate accuracy
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)

# STEP 8 — Save model
joblib.dump(model, "../model.pkl")

print("Model saved as model.pkl")

joblib.dump(feature_columns,"../columns.pkl")
print("Model and columns saved.")

preds = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test,preds))

plot_importance(model)
plt.show()