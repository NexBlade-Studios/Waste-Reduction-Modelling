import pandas as pd
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

r,p = np.load("./results/parameter.npy")

print("r =", r)
print("p =", p)

simulation = nbinom.rvs(r, p, size=10000)

counts = demand.value_counts().sort_index()

print("Simulated mean:", np.mean(simulation))
print("Simulated variance:", np.var(simulation))

np.save("./results/simulation.npy", simulation)