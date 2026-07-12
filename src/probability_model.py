import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import nbinom

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

simulation = nbinom.rvs(r, p, size=10000)

counts = demand.value_counts().sort_index()

plt.figure(figsize=(10,5))
plt.bar(counts.index, counts.values/len(demand))

plt.xlabel("Orders")
plt.ylabel("Probability")
plt.title("Actual Demand Distribution")

plt.show()

actual = demand.value_counts().sort_index() / len(demand)

sim = pd.Series(simulation).value_counts().sort_index() / len(simulation)

plt.figure(figsize=(10,5))

plt.bar(actual.index, actual.values, alpha=0.6, label="Actual demand")

plt.plot(sim.index, sim.values, color="red", linewidth=2, label="Negative Binomial model")

plt.xlabel("Orders")
plt.ylabel("Probability")
plt.title("Actual vs Modelled Demand")
plt.legend()

plt.show()

results = []

for supply in range(5, 60):
    waste = np.maximum(supply - simulation, 0)
    shortage = np.maximum(simulation - supply, 0)

    cost = np.mean(waste) + 5 * np.mean(shortage)

    results.append([supply, cost])

results = np.array(results)

best_index = np.argmin(results[:, 1])

best_supply = results[best_index, 0]
best_cost = results[best_index, 1]

print("Optimal supply:", best_supply)
print("Minimum expected cost:", best_cost)

plt.figure(figsize=(10,5))

plt.plot(results[:,0], results[:,1])

plt.xlabel("Supply Level")
plt.ylabel("Expected Cost")
plt.title("Optimising Food Supply")

plt.grid(True)

plt.show()