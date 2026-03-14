import pandas as pd
import joblib

print("Camp Triage AI System")

# load model
model = joblib.load("model.pkl")

# load training columns
model_columns = joblib.load("columns.pkl")

# get inputs
name = input("Name: ")
age = int(input("Age: "))
weight = int(input("Weight: "))
heartrate = int(input("Heart Rate: "))
radiation = int(input("Radiation Level: "))
injury = int(input("Injury Level: "))
hydration = int(input("Hydration Level: "))

# create dataframe
data = pd.DataFrame([{
    "Name": name,
    "Age": age,
    "Weight": weight,
    "HeartRate": heartrate,
    "Radiation": radiation,
    "InjuryLevel": injury,
    "Hydration": hydration
}])

# encode
data = pd.get_dummies(data)

# align with training columns
data = data.reindex(columns=model_columns, fill_value=0)

# predict
prediction = model.predict(data)

print("\nPredicted Calories Needed:", int(prediction[0]))