import numpy as np
import matplotlib.pyplot as plt

#load sim data
simulation = np.load("./results/simulation.npy")

results = []

for supply in range(5,60):
    waste = np.maximum(
        supply - simulation,
        0
    )
    shortage = np.maximum(
        simulation - supply,
        0
    )

    cost = np.mean(waste) + 5*np.mean(shortage)

    results.append(
        [supply, cost]
    )

results = np.array(results)

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