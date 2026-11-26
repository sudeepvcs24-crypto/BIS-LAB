RESEARCH PAPER:https://ieeexplore.ieee.org/document/488968
```
PROBLEM STATEMENT:

Implement the Particle Swarm Optimization (PSO) algorithm using Python.
Define a mathematical function to be optimized.
Initialize particles with random positions and velocities.
Evaluate the fitness of each particle.
Update each particle's velocity and position using its personal best and global best.
Repeat the process for a fixed number of iterations until convergence.
Display the best solution and optimal fitness value.
Plot graphs to visualize the best solution and the convergence of the algorithm.
```
CODE IMPLEMENTATION
import numpy as np
import matplotlib.pyplot as plt

# Objective function
def objective(x):
    return -(x - 3)**2 + 10

# PSO Parameters
num_particles = 30
num_iterations = 100

lower_bound = -10
upper_bound = 10

w = 0.7
c1 = 1.5
c2 = 1.5

np.random.seed(42)

# Initialize particles
positions = np.random.uniform(
    lower_bound, upper_bound, num_particles
)

velocities = np.random.uniform(-1, 1, num_particles)

# Personal best
pbest_positions = positions.copy()
pbest_fitness = objective(pbest_positions)

# Global best
gbest_index = np.argmax(pbest_fitness)

gbest_position = pbest_positions[gbest_index]
gbest_fitness = pbest_fitness[gbest_index]

best_fitness_history = []

# Main PSO loop
for iteration in range(1, num_iterations + 1):

    for i in range(num_particles):

        r1 = np.random.random()
        r2 = np.random.random()

        # Update velocity
        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (pbest_positions[i] - positions[i])
            + c2 * r2 * (gbest_position - positions[i])
        )

        # Update position
        positions[i] += velocities[i]

        positions[i] = np.clip(
            positions[i], lower_bound, upper_bound
        )

        # Calculate fitness
        current_fitness = objective(positions[i])

        # Update personal best
        if current_fitness > pbest_fitness[i]:

            pbest_positions[i] = positions[i]
            pbest_fitness[i] = current_fitness

    # Update global best
    gbest_index = np.argmax(pbest_fitness)

    if pbest_fitness[gbest_index] > gbest_fitness:

        gbest_position = pbest_positions[gbest_index]
        gbest_fitness = pbest_fitness[gbest_index]

    # Store best fitness
    best_fitness_history.append(gbest_fitness)

    # Print every iteration
    print(
        f"Iteration: {iteration} "
        f"Best Fitness: {gbest_fitness:.6f}"
    )

# Final output
print("\nBest Position:", round(float(gbest_position), 6))
print("Maximum Fitness:", round(float(gbest_fitness), 6))


# GRAPH 1: Objective Function
x_values = np.linspace(lower_bound, upper_bound, 400)
y_values = objective(x_values)

plt.figure(figsize=(8, 5))

plt.plot(x_values, y_values, label="Objective Function")

plt.scatter(
    [gbest_position],
    [gbest_fitness],
    color="red",
    label="PSO Best Solution"
)

plt.title("PSO - Best Solution on Objective Function")
plt.xlabel("x")
plt.ylabel("Fitness f(x)")
plt.grid(True)
plt.legend()
plt.show()


# GRAPH 2: Convergence Graph
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, num_iterations + 1),
    best_fitness_history,
    marker="."
)

plt.title("PSO Convergence Graph")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.show()

Iteration: 1 Best Fitness: 9.995996
Iteration: 2 Best Fitness: 9.999620
Iteration: 3 Best Fitness: 9.999620
Iteration: 4 Best Fitness: 9.999620
Iteration: 5 Best Fitness: 9.999620
Iteration: 6 Best Fitness: 9.999620
Iteration: 7 Best Fitness: 9.999620
Iteration: 8 Best Fitness: 9.999620
Iteration: 9 Best Fitness: 9.999665
Iteration: 10 Best Fitness: 9.999665
Iteration: 11 Best Fitness: 9.999665
Iteration: 12 Best Fitness: 9.999849
Iteration: 13 Best Fitness: 9.999849
Iteration: 14 Best Fitness: 9.999849
Iteration: 15 Best Fitness: 9.999849
Iteration: 16 Best Fitness: 9.999849
Iteration: 17 Best Fitness: 9.999942
Iteration: 18 Best Fitness: 9.999997
Iteration: 19 Best Fitness: 9.999997
Iteration: 20 Best Fitness: 9.999997
Iteration: 21 Best Fitness: 9.999999
Iteration: 22 Best Fitness: 9.999999
Iteration: 23 Best Fitness: 9.999999
Iteration: 24 Best Fitness: 9.999999
Iteration: 25 Best Fitness: 9.999999
Iteration: 26 Best Fitness: 10.000000
Iteration: 27 Best Fitness: 10.000000
Iteration: 28 Best Fitness: 10.000000
Iteration: 29 Best Fitness: 10.000000
Iteration: 30 Best Fitness: 10.000000
Iteration: 31 Best Fitness: 10.000000
Iteration: 32 Best Fitness: 10.000000
Iteration: 33 Best Fitness: 10.000000
Iteration: 34 Best Fitness: 10.000000
Iteration: 35 Best Fitness: 10.000000
Iteration: 36 Best Fitness: 10.000000
Iteration: 37 Best Fitness: 10.000000
Iteration: 38 Best Fitness: 10.000000
Iteration: 39 Best Fitness: 10.000000
Iteration: 40 Best Fitness: 10.000000
Iteration: 41 Best Fitness: 10.000000
Iteration: 42 Best Fitness: 10.000000
Iteration: 43 Best Fitness: 10.000000
Iteration: 44 Best Fitness: 10.000000
Iteration: 45 Best Fitness: 10.000000
Iteration: 46 Best Fitness: 10.000000
Iteration: 47 Best Fitness: 10.000000
Iteration: 48 Best Fitness: 10.000000
Iteration: 49 Best Fitness: 10.000000
Iteration: 50 Best Fitness: 10.000000
Iteration: 51 Best Fitness: 10.000000
Iteration: 52 Best Fitness: 10.000000
Iteration: 53 Best Fitness: 10.000000
Iteration: 54 Best Fitness: 10.000000
Iteration: 55 Best Fitness: 10.000000
Iteration: 56 Best Fitness: 10.000000
Iteration: 57 Best Fitness: 10.000000
Iteration: 58 Best Fitness: 10.000000
Iteration: 59 Best Fitness: 10.000000
Iteration: 60 Best Fitness: 10.000000
Iteration: 61 Best Fitness: 10.000000
Iteration: 62 Best Fitness: 10.000000
Iteration: 63 Best Fitness: 10.000000
Iteration: 64 Best Fitness: 10.000000
Iteration: 65 Best Fitness: 10.000000
Iteration: 66 Best Fitness: 10.000000
Iteration: 67 Best Fitness: 10.000000
Iteration: 68 Best Fitness: 10.000000
Iteration: 69 Best Fitness: 10.000000
Iteration: 70 Best Fitness: 10.000000
Iteration: 71 Best Fitness: 10.000000
Iteration: 72 Best Fitness: 10.000000
Iteration: 73 Best Fitness: 10.000000
Iteration: 74 Best Fitness: 10.000000
Iteration: 75 Best Fitness: 10.000000
Iteration: 76 Best Fitness: 10.000000
Iteration: 77 Best Fitness: 10.000000
Iteration: 78 Best Fitness: 10.000000
Iteration: 79 Best Fitness: 10.000000
Iteration: 80 Best Fitness: 10.000000
Iteration: 81 Best Fitness: 10.000000
Iteration: 82 Best Fitness: 10.000000
Iteration: 83 Best Fitness: 10.000000
Iteration: 84 Best Fitness: 10.000000
Iteration: 85 Best Fitness: 10.000000
Iteration: 86 Best Fitness: 10.000000
Iteration: 87 Best Fitness: 10.000000
Iteration: 88 Best Fitness: 10.000000
Iteration: 89 Best Fitness: 10.000000
Iteration: 90 Best Fitness: 10.000000
Iteration: 91 Best Fitness: 10.000000
Iteration: 92 Best Fitness: 10.000000
Iteration: 93 Best Fitness: 10.000000
Iteration: 94 Best Fitness: 10.000000
Iteration: 95 Best Fitness: 10.000000
Iteration: 96 Best Fitness: 10.000000
Iteration: 97 Best Fitness: 10.000000
Iteration: 98 Best Fitness: 10.000000
Iteration: 99 Best Fitness: 10.000000
Iteration: 100 Best Fitness: 10.000000

Best Position: 3.0
Maximum Fitness: 10.0

