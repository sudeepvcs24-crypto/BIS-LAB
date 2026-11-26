RESEARCH PAPER:https://arxiv.org/abs/cs/0102027
```
PROBLEM STATEMENT
1. Implement a Gene Expression Algorithm using Python.
2. Define a mathematical function to be optimized.
3. Initialize a population of random genetic sequences.
4. Express each genetic sequence into a numerical solution.
5. Evaluate the fitness of each expressed solution.
6. Select genetic sequences based on their fitness.
7. Perform crossover and mutation to generate offspring.
8. Repeat the process for a fixed number of generations.
9. Track and display the best solution found during the iterations.
10. Generate graphs to visualize the best solution and convergence.
```
CODE IMPLEMENTATION
```PY
import numpy as np
import matplotlib.pyplot as plt

# Gene Expression Algorithm for Function Optimization
# Objective: minimize f(x) = (x - 3)^2

def objective(x):
    return (x - 3) ** 2

# Parameters
population_size = 50
gene_length = 24
mutation_rate = 0.02
crossover_rate = 0.8
num_generations = 100

lower_bound = -10
upper_bound = 10

np.random.seed(42)

# Initialize population of binary chromosomes
population = np.random.randint(
    0, 2, (population_size, gene_length)
)

# Gene expression: binary chromosome -> numerical solution
def gene_expression(chromosome):
    value = int("".join(map(str, chromosome)), 2)
    max_value = (2 ** gene_length) - 1

    normalized = value / max_value

    return lower_bound + normalized * (
        upper_bound - lower_bound
    )

# Evaluate population
def evaluate_population(pop):
    expressed = np.array([
        gene_expression(ch) for ch in pop
    ])

    fitness = objective(expressed)

    return expressed, fitness

# Tournament selection
def tournament_selection(pop, fit, tournament_size=3):
    candidates = np.random.choice(
        len(pop), tournament_size, replace=False
    )

    winner = candidates[np.argmin(fit[candidates])]

    return pop[winner].copy()

# Initial evaluation
expressed, fitness = evaluate_population(population)

best_idx = np.argmin(fitness)

best_chromosome = population[best_idx].copy()
best_x = float(expressed[best_idx])
best_fitness = float(fitness[best_idx])

history = []

# Evolution process
for generation in range(1, num_generations + 1):

    new_population = [
        best_chromosome.copy()
    ]  # Elitism

    while len(new_population) < population_size:

        # Selection
        parent1 = tournament_selection(
            population, fitness
        )

        parent2 = tournament_selection(
            population, fitness
        )

        # Crossover
        if np.random.random() < crossover_rate:

            point = np.random.randint(
                1, gene_length
            )

            child1 = np.concatenate(
                (parent1[:point], parent2[point:])
            )

            child2 = np.concatenate(
                (parent2[:point], parent1[point:])
            )

        else:
            child1 = parent1.copy()
            child2 = parent2.copy()

        # Mutation
        for child in (child1, child2):

            mutation_mask = (
                np.random.random(gene_length)
                < mutation_rate
            )

            child[mutation_mask] = (
                1 - child[mutation_mask]
            )

        new_population.extend([child1, child2])

    # Replace old population
    population = np.array(
        new_population[:population_size]
    )

    # Gene expression and fitness evaluation
    expressed, fitness = evaluate_population(
        population
    )

    # Find current best
    current_idx = np.argmin(fitness)

    # Update best solution
    if fitness[current_idx] < best_fitness:

        best_chromosome = (
            population[current_idx].copy()
        )

        best_x = float(
            expressed[current_idx]
        )

        best_fitness = float(
            fitness[current_idx]
        )

    history.append(best_fitness)

    # Iteration-wise output
    print(
        f"Iteration: {generation} "
        f"Best Fitness: {best_fitness:.6f}"
    )

# Final result
print("\nBest Expressed Solution (x):", round(best_x, 6))
print("Minimum Fitness:", round(best_fitness, 8))

print(
    "Best Gene Sequence:",
    "".join(map(str, best_chromosome))
)

# GRAPH 1: Objective Function and Best Solution
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
    [best_x], [best_fitness],
    label="GEA Best Solution",
    zorder=3
)

plt.title("Gene Expression Algorithm - Best Solution")
plt.xlabel("Expressed x")
plt.ylabel("Fitness f(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# GRAPH 2: Convergence Graph
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, num_generations + 1),
    history,
    marker="."
)

plt.title("Gene Expression Algorithm Convergence")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.tight_layout()
plt.show()
```
```
Iteration: 1 Best Fitness: 0.001378
Iteration: 2 Best Fitness: 0.000069
Iteration: 3 Best Fitness: 0.000069
Iteration: 4 Best Fitness: 0.000069
Iteration: 5 Best Fitness: 0.000064
Iteration: 6 Best Fitness: 0.000015
Iteration: 7 Best Fitness: 0.000007
Iteration: 8 Best Fitness: 0.000007
Iteration: 9 Best Fitness: 0.000007
Iteration: 10 Best Fitness: 0.000004
Iteration: 11 Best Fitness: 0.000004
Iteration: 12 Best Fitness: 0.000004
Iteration: 13 Best Fitness: 0.000004
Iteration: 14 Best Fitness: 0.000004
Iteration: 15 Best Fitness: 0.000004
Iteration: 16 Best Fitness: 0.000004
Iteration: 17 Best Fitness: 0.000004
Iteration: 18 Best Fitness: 0.000004
Iteration: 19 Best Fitness: 0.000004
Iteration: 20 Best Fitness: 0.000004
Iteration: 21 Best Fitness: 0.000004
Iteration: 22 Best Fitness: 0.000004
Iteration: 23 Best Fitness: 0.000004
Iteration: 24 Best Fitness: 0.000004
Iteration: 25 Best Fitness: 0.000004
Iteration: 26 Best Fitness: 0.000004
Iteration: 27 Best Fitness: 0.000004
Iteration: 28 Best Fitness: 0.000004
Iteration: 29 Best Fitness: 0.000004
Iteration: 30 Best Fitness: 0.000004
Iteration: 31 Best Fitness: 0.000004
Iteration: 32 Best Fitness: 0.000004
Iteration: 33 Best Fitness: 0.000004
Iteration: 34 Best Fitness: 0.000004
Iteration: 35 Best Fitness: 0.000004
Iteration: 36 Best Fitness: 0.000004
Iteration: 37 Best Fitness: 0.000004
Iteration: 38 Best Fitness: 0.000004
Iteration: 39 Best Fitness: 0.000004
Iteration: 40 Best Fitness: 0.000004
Iteration: 41 Best Fitness: 0.000004
Iteration: 42 Best Fitness: 0.000004
Iteration: 43 Best Fitness: 0.000004
Iteration: 44 Best Fitness: 0.000004
Iteration: 45 Best Fitness: 0.000004
Iteration: 46 Best Fitness: 0.000004
Iteration: 47 Best Fitness: 0.000004
Iteration: 48 Best Fitness: 0.000004
Iteration: 49 Best Fitness: 0.000004
Iteration: 50 Best Fitness: 0.000004
Iteration: 51 Best Fitness: 0.000004
Iteration: 52 Best Fitness: 0.000004
Iteration: 53 Best Fitness: 0.000004
Iteration: 54 Best Fitness: 0.000004
Iteration: 55 Best Fitness: 0.000004
Iteration: 56 Best Fitness: 0.000004
Iteration: 57 Best Fitness: 0.000004
Iteration: 58 Best Fitness: 0.000004
Iteration: 59 Best Fitness: 0.000004
Iteration: 60 Best Fitness: 0.000004
Iteration: 61 Best Fitness: 0.000004
Iteration: 62 Best Fitness: 0.000004
Iteration: 63 Best Fitness: 0.000004
Iteration: 64 Best Fitness: 0.000004
Iteration: 65 Best Fitness: 0.000004
Iteration: 66 Best Fitness: 0.000004
Iteration: 67 Best Fitness: 0.000004
Iteration: 68 Best Fitness: 0.000004
Iteration: 69 Best Fitness: 0.000004
Iteration: 70 Best Fitness: 0.000004
Iteration: 71 Best Fitness: 0.000004
Iteration: 72 Best Fitness: 0.000004
Iteration: 73 Best Fitness: 0.000004
Iteration: 74 Best Fitness: 0.000004
Iteration: 75 Best Fitness: 0.000004
Iteration: 76 Best Fitness: 0.000004
Iteration: 77 Best Fitness: 0.000004
Iteration: 78 Best Fitness: 0.000004
Iteration: 79 Best Fitness: 0.000004
Iteration: 80 Best Fitness: 0.000004
Iteration: 81 Best Fitness: 0.000004
Iteration: 82 Best Fitness: 0.000004
Iteration: 83 Best Fitness: 0.000004
Iteration: 84 Best Fitness: 0.000004
Iteration: 85 Best Fitness: 0.000004
Iteration: 86 Best Fitness: 0.000004
Iteration: 87 Best Fitness: 0.000004
Iteration: 88 Best Fitness: 0.000004
Iteration: 89 Best Fitness: 0.000004
Iteration: 90 Best Fitness: 0.000004
Iteration: 91 Best Fitness: 0.000004
Iteration: 92 Best Fitness: 0.000004
Iteration: 93 Best Fitness: 0.000004
Iteration: 94 Best Fitness: 0.000004
Iteration: 95 Best Fitness: 0.000004
Iteration: 96 Best Fitness: 0.000004
Iteration: 97 Best Fitness: 0.000004
Iteration: 98 Best Fitness: 0.000004
Iteration: 99 Best Fitness: 0.000004
Iteration: 100 Best Fitness: 0.000004

Best Expressed Solution (x): 2.998046
Minimum Fitness: 3.82e-06
Best Gene Sequence: 101001100101111111111111
```
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/0c4446b4-6f63-4534-90f4-4fa84434f8e8" />
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/afeb101b-8bc8-4be5-ace3-e8ca8fceb2ef" />



