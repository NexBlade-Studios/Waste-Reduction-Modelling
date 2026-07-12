import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv("./data/train.csv")

# limits the data to an arbitrary meal type and center
data = train[
    (train["meal_id"] == 2139) &
    (train["center_id"] == 55)
]

# checks if there is sufficient data points to use
print("Length is", len(data))

demand = data["num_orders"]

# plots new data
plt.figure(figsize=(10,5))
plt.plot(data["num_orders"])
plt.xlabel("Observation")
plt.ylabel("Orders")
plt.title("Chosen Meal and Centre")

plt.show()

# finds mean and variance of new data
mean = np.mean(demand)
variance = np.var(demand, ddof=1)

print("Mean:", mean)
print("Variance:", variance)

# r is the number of successes
r = (mean** 2) / (variance - mean)

# p is the probability
p = r / (r + mean)

print("r =", r)
print("p =", p)

np.save("./results/parameters.npy", [r,p])