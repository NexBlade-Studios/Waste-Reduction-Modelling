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

demand = data["num_orders"]

# finds mean and variance of new data
mean = np.mean(demand)
variance = np.var(demand, ddof=1)

print("Mean:", mean)
print("Variance:", variance)

r,p = np.load("./results/parameters.npy")

print("r =", r)
print("p =", p)

simulation = nbinom.rvs(r, p, size=10000)

counts = demand.value_counts().sort_index()

print("Simulated mean:", np.mean(simulation))
print("Simulated variance:", np.var(simulation))

np.save("./results/simulation.npy", simulation)

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