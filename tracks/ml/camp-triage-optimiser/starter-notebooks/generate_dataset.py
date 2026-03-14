import pandas as pd
import random

names = ["Aaron","Adam","Alex","Ben","Chris","Daniel","Evan","Frank","George","Henry"]

data = []

for i in range(500):

    name = random.choice(names)
    age = random.randint(18,60)
    weight = random.randint(50,100)
    heart = random.randint(60,120)
    radiation = random.randint(0,10)
    injury = random.randint(0,5)
    hydration = random.randint(1,5)

    calories = 1800 + weight*5 + injury*100 + radiation*50 - hydration*30

    data.append([
        name,
        age,
        weight,
        heart,
        radiation,
        injury,
        hydration,
        calories
    ])

df = pd.DataFrame(data, columns=[
    "Name",
    "Age",
    "Weight",
    "HeartRate",
    "Radiation",
    "InjuryLevel",
    "Hydration",
    "CaloriesNeeded"
])

df.to_csv("dataset.csv", index=False)

print("Dataset generated with 500 rows")