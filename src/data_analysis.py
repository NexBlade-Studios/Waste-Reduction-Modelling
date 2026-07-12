import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("./data/train.csv")

print(train.head())
print(train.info())
print(train.describe())

plt.figure(figsize=(10,5))
plt.plot(train["num_orders"])
plt.xlabel("Observation")
plt.ylabel("Orders")
plt.title("Historical Food Demand")

plt.show()
