RESEARCH PAPER:https://surface.syr.edu/eecs_techreports/167/
```
PROBLEM STATEMENT
1. Implement a Parallel Cellular Algorithm using Python.
2. Define a mathematical function to be optimized.
3. Initialize a grid of cells with random candidate solutions.
4. Evaluate the fitness of each cell.
5. Update each cell based on its neighboring cells using predefined rules.
6. Perform cell updates simultaneously in each iteration.
7. Repeat the process for a fixed number of iterations.
8. Track and display the best solution found during the iterations.
9. Generate graphs to visualize the best solution and algorithm convergence.
```
CODE IMPLEMENTATION
```PY
import numpy as np
import matplotlib.pyplot as plt

# Parallel Cellular Algorithm for Function Optimization

def objective(x):
    return (x - 3) ** 2

# Parameters
grid_size = 20
num_iterations = 100
lower_bound = -10
upper_bound = 10
perturbation_scale = 0.5

np.random.seed(42)

# Initialize cells randomly
cells = np.random.uniform(
    lower_bound, upper_bound,
    (grid_size, grid_size)
)

fitness = objective(cells)

# Find initial best solution
best_idx = np.unravel_index(
    np.argmin(fitness), fitness.shape
)

best_position = float(cells[best_idx])
best_fitness = float(fitness[best_idx])

best_fitness_history = []

# Main iteration loop
for iteration in range(1, num_iterations + 1):

    # Create copy for synchronous updates
    next_cells = cells.copy()

    for row in range(grid_size):
        for col in range(grid_size):

            # Von Neumann neighborhood
            neighbors = [
                (row, col),
                ((row - 1) % grid_size, col),
                ((row + 1) % grid_size, col),
                (row, (col - 1) % grid_size),
                (row, (col + 1) % grid_size)
            ]

            # Get neighborhood values
            neighbor_values = np.array([
                cells[r, c] for r, c in neighbors
            ])

            # Evaluate neighboring cells
            neighbor_fitness = objective(
                neighbor_values
            )

            # Select local best
            local_best = neighbor_values[
                np.argmin(neighbor_fitness)
            ]

            # Generate new candidate
            candidate = local_best + np.random.normal(
                0, perturbation_scale
            )

            candidate = np.clip(
                candidate, lower_bound, upper_bound
            )

            # Replace cell if candidate is better
            if objective(candidate) < objective(cells[row, col]):
                next_cells[row, col] = candidate

    # Apply all updates simultaneously
    cells = next_cells

    # Evaluate new population
    fitness = objective(cells)

    # Find current best
    current_idx = np.unravel_index(
        np.argmin(fitness), fitness.shape
    )

    current_best = float(cells[current_idx])
    current_best_fitness = float(fitness[current_idx])

    # Update global best
    if current_best_fitness < best_fitness:
        best_position = current_best
        best_fitness = current_best_fitness

    best_fitness_history.append(best_fitness)

    # Print every iteration
    print(
        f"Iteration: {iteration} "
        f"Best Fitness: {best_fitness:.6f}"
    )

# Final output
print("\nBest Solution (x):", round(best_position, 6))
print("Minimum Fitness:", round(best_fitness, 6))


# GRAPH 1: Objective Function
x_values = np.linspace(
    lower_bound, upper_bound, 400
)

y_values = objective(x_values)

plt.figure(figsize=(8, 5))

plt.plot(
    x_values, y_values,
    label="Objective Function"
)

plt.scatter(
    [best_position], [best_fitness],
    label="Best Solution", zorder=3
)

plt.title("Parallel Cellular Algorithm - Best Solution")
plt.xlabel("x")
plt.ylabel("Fitness f(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# GRAPH 2: Convergence Graph
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, num_iterations + 1),
    best_fitness_history,
    marker="."
)

plt.title("Parallel Cellular Algorithm Convergence")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.tight_layout()
plt.show()
```
```
Iteration: 1 Best Fitness: 0.000001
Iteration: 2 Best Fitness: 0.000001
Iteration: 3 Best Fitness: 0.000001
Iteration: 4 Best Fitness: 0.000001
Iteration: 5 Best Fitness: 0.000001
Iteration: 6 Best Fitness: 0.000001
Iteration: 7 Best Fitness: 0.000001
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

Best Solution (x): 2.999961
Minimum Fitness: 0.0
```
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/d4e62c60-d1e0-4876-8794-487477f4ceba" />
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/ad6fecbd-a658-4e12-9d2e-4568c5b017e5" />




