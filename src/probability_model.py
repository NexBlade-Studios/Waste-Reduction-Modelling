import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

train = pd.read_csv("./data/train.csv")

data = train[
    (train["meal_id"] == 2139) &
    (train["center_id"] == 55)
]

demand = data["num_orders"]

plt.figure(figsize=(10,5))
plt.plot(data["num_orders"])
plt.xlabel("Observation")
plt.ylabel("Orders")
plt.title("Chosen Meal and Centre")

plt.show()


mean = np.mean(demand)
variance = np.var(demand)

print("Mean:", mean)
print("Variance:", variance)