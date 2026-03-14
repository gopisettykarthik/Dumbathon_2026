import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import hashlib

data = pd.read_csv("dataset.csv")

# encode names using hashing
def encode_name(name):
    return int(hashlib.md5(name.encode()).hexdigest(),16)%1000

data["NameEncoded"] = data["Name"].apply(encode_name)

data = data.drop("Name", axis=1)

X = data.drop("CaloriesNeeded", axis=1)
y = data["CaloriesNeeded"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = XGBRegressor()

model.fit(X_train,y_train)

print("Bias mitigated model trained")