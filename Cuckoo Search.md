RESEARCH PAPER:https://ieeexplore.ieee.org/abstract/document/5393690
```
PROBLEM STATEMENT:
Implement the Cuckoo Search (CS) algorithm using Python.
Define a mathematical function to be optimized.
Initialize nests with random positions.
Evaluate the fitness of each nest.
Generate new solutions using Lévy flights.
Replace nests when better solutions are found.
Abandon a fraction of the worst nests and replace them with random solutions.
Repeat the process for a fixed number of iterations.
Display the best solution and its fitness.
Plot graphs to visualize the best solution and convergence.
```

CODE IMPLEMENTATION

```PY
import numpy as np
import matplotlib.pyplot as plt
import math

# Cuckoo Search (CS) for Function Optimization
# Objective: minimize f(x) = (x - 3)^2
# The theoretical minimum is x = 3, f(x) = 0.

def objective(x):
    return (x - 3) ** 2

# Parameters
num_nests = 25
num_iterations = 100
discovery_probability = 0.25  # pa
lower_bound = -10
upper_bound = 10
step_size = 0.01
beta = 1.5

np.random.seed(42)

# Mantegna method for Levy-flight steps
def levy_flight(beta=1.5):
    sigma_u = (
        math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta
           * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)

    u = np.random.normal(0, sigma_u)
    v = np.random.normal(0, 1)
    step = u / (abs(v) ** (1 / beta) + 1e-10)
    return step

# Initialize nests randomly
nests = np.random.uniform(lower_bound, upper_bound, num_nests)
fitness = objective(nests)

best_index = np.argmin(fitness)
best_nest = nests[best_index]
best_fitness = fitness[best_index]

best_fitness_history = []
best_position_history = []

# Main Cuckoo Search loop
for iteration in range(1, num_iterations + 1):

    # Generate new solutions using Levy flights
    for i in range(num_nests):
        new_nest = nests[i] + step_size * levy_flight(beta)
        new_nest = np.clip(new_nest, lower_bound, upper_bound)
        new_fitness = objective(new_nest)

        # Choose a random nest; replace if new solution is better
        j = np.random.randint(num_nests)
        if new_fitness < fitness[j]:
            nests[j] = new_nest
            fitness[j] = new_fitness

    # Abandon a fraction of worst nests and replace them randomly
    number_to_abandon = max(1, int(discovery_probability * num_nests))
    worst_indices = np.argsort(fitness)[-number_to_abandon:]

    for idx in worst_indices:
        nests[idx] = np.random.uniform(lower_bound, upper_bound)
        fitness[idx] = objective(nests[idx])

    # Update best solution found so far
    current_best_index = np.argmin(fitness)
    if fitness[current_best_index] < best_fitness:
        best_nest = nests[current_best_index]
        best_fitness = fitness[current_best_index]

    best_fitness_history.append(best_fitness)
    best_position_history.append(best_nest)

    # Print each iteration, as requested
    print(f"Iteration: {iteration} Best Fitness: {best_fitness:.6f}")

# Final result
print("\\nBest Solution (x):", round(float(best_nest), 6))
print("Minimum Fitness:", round(float(best_fitness), 6))

# Graph 1: Objective function and best solution
x_values = np.linspace(lower_bound, upper_bound, 400)
y_values = objective(x_values)

plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, label="Objective Function")
plt.scatter([best_nest], [best_fitness], color="red",
            label="CS Best Solution", zorder=3)
plt.title("Cuckoo Search - Best Solution")
plt.xlabel("x")
plt.ylabel("Fitness f(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Graph 2: Convergence graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_iterations + 1), best_fitness_history, marker=".")
plt.title("Cuckoo Search Convergence Graph")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.tight_layout()
plt.show()
```
```
Iteration: 1 Best Fitness: 0.073871
Iteration: 2 Best Fitness: 0.073871
Iteration: 3 Best Fitness: 0.069852
Iteration: 4 Best Fitness: 0.000093
Iteration: 5 Best Fitness: 0.000093
Iteration: 6 Best Fitness: 0.000004
Iteration: 7 Best Fitness: 0.000003
Iteration: 8 Best Fitness: 0.000001
Iteration: 9 Best Fitness: 0.000000
Iteration: 10 Best Fitness: 0.000000
Iteration: 11 Best Fitness: 0.000000
Iteration: 12 Best Fitness: 0.000000
Iteration: 13 Best Fitness: 0.000000
Iteration: 14 Best Fitness: 0.000000
Iteration: 15 Best Fitness: 0.000000
Iteration: 16 Best Fitness: 0.000000
Iteration: 17 Best Fitness: 0.000000
Iteration: 18 Best Fitness: 0.000000
Iteration: 19 Best Fitness: 0.000000
Iteration: 20 Best Fitness: 0.000000
Iteration: 21 Best Fitness: 0.000000
Iteration: 22 Best Fitness: 0.000000
Iteration: 23 Best Fitness: 0.000000
Iteration: 24 Best Fitness: 0.000000
Iteration: 25 Best Fitness: 0.000000
Iteration: 26 Best Fitness: 0.000000
Iteration: 27 Best Fitness: 0.000000
Iteration: 28 Best Fitness: 0.000000
Iteration: 29 Best Fitness: 0.000000
Iteration: 30 Best Fitness: 0.000000
Iteration: 31 Best Fitness: 0.000000
Iteration: 32 Best Fitness: 0.000000
Iteration: 33 Best Fitness: 0.000000
Iteration: 34 Best Fitness: 0.000000
Iteration: 35 Best Fitness: 0.000000
Iteration: 36 Best Fitness: 0.000000
Iteration: 37 Best Fitness: 0.000000
Iteration: 38 Best Fitness: 0.000000
Iteration: 39 Best Fitness: 0.000000
Iteration: 40 Best Fitness: 0.000000
Iteration: 41 Best Fitness: 0.000000
Iteration: 42 Best Fitness: 0.000000
Iteration: 43 Best Fitness: 0.000000
Iteration: 44 Best Fitness: 0.000000
Iteration: 45 Best Fitness: 0.000000
Iteration: 46 Best Fitness: 0.000000
Iteration: 47 Best Fitness: 0.000000
Iteration: 48 Best Fitness: 0.000000
Iteration: 49 Best Fitness: 0.000000
Iteration: 50 Best Fitness: 0.000000
Iteration: 51 Best Fitness: 0.000000
Iteration: 52 Best Fitness: 0.000000
Iteration: 53 Best Fitness: 0.000000
Iteration: 54 Best Fitness: 0.000000
Iteration: 55 Best Fitness: 0.000000
Iteration: 56 Best Fitness: 0.000000
Iteration: 57 Best Fitness: 0.000000
Iteration: 58 Best Fitness: 0.000000
Iteration: 59 Best Fitness: 0.000000
Iteration: 60 Best Fitness: 0.000000
Iteration: 61 Best Fitness: 0.000000
Iteration: 62 Best Fitness: 0.000000
Iteration: 63 Best Fitness: 0.000000
Iteration: 64 Best Fitness: 0.000000
Iteration: 65 Best Fitness: 0.000000
Iteration: 66 Best Fitness: 0.000000
Iteration: 67 Best Fitness: 0.000000
Iteration: 68 Best Fitness: 0.000000
Iteration: 69 Best Fitness: 0.000000
Iteration: 70 Best Fitness: 0.000000
Iteration: 71 Best Fitness: 0.000000
Iteration: 72 Best Fitness: 0.000000
Iteration: 73 Best Fitness: 0.000000
Iteration: 74 Best Fitness: 0.000000
Iteration: 75 Best Fitness: 0.000000
Iteration: 76 Best Fitness: 0.000000
Iteration: 77 Best Fitness: 0.000000
Iteration: 78 Best Fitness: 0.000000
Iteration: 79 Best Fitness: 0.000000
Iteration: 80 Best Fitness: 0.000000
Iteration: 81 Best Fitness: 0.000000
Iteration: 82 Best Fitness: 0.000000
Iteration: 83 Best Fitness: 0.000000
Iteration: 84 Best Fitness: 0.000000
Iteration: 85 Best Fitness: 0.000000
Iteration: 86 Best Fitness: 0.000000
Iteration: 87 Best Fitness: 0.000000
Iteration: 88 Best Fitness: 0.000000
Iteration: 89 Best Fitness: 0.000000
Iteration: 90 Best Fitness: 0.000000
Iteration: 91 Best Fitness: 0.000000
Iteration: 92 Best Fitness: 0.000000
Iteration: 93 Best Fitness: 0.000000
Iteration: 94 Best Fitness: 0.000000
Iteration: 95 Best Fitness: 0.000000
Iteration: 96 Best Fitness: 0.000000
Iteration: 97 Best Fitness: 0.000000
Iteration: 98 Best Fitness: 0.000000
Iteration: 99 Best Fitness: 0.000000
Iteration: 100 Best Fitness: 0.000000
\nBest Solution (x): 2.999985
Minimum Fitness: 0.0
```
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/de523c5f-48ef-47d4-9493-b8cc64613508" />
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/d3865783-9674-4751-823b-9e0e5d2e567c" />


