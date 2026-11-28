RESEARCH PAPER : https://ieeexplore.ieee.org/abstract/document/484436/
```
PROBLEM STATEMENT:
Implement the Ant Colony Optimization (ACO) algorithm using Python to solve the Traveling Salesman Problem (TSP). Given a set of cities and their coordinates, find the shortest possible route that visits each city exactly once and returns to the starting city. The algorithm uses pheromone trails and heuristic information to guide ants toward better routes.
- Input: A set of cities with coordinates.
- Objective: Find the shortest tour.
- Constraint: Each city must be visited exactly once.
- Return: The tour must return to the starting city.
- Output: Best route and minimum total distance.
```

CODE IMPLEMENTATION
```PY
import numpy as np
import matplotlib.pyplot as plt
import random

# -------------------- ACO for Traveling Salesman Problem --------------------

# City coordinates
cities = np.array([
    [0, 0], [2, 6], [5, 2], [6, 6], [8, 3],
    [1, 4], [3, 8], [7, 7], [9, 6], [4, 1]
])

num_cities = len(cities)
num_ants = 20
num_iterations = 100

alpha = 1.0
beta = 3.0
rho = 0.5
Q = 100.0

random.seed(42)
np.random.seed(42)

# Calculate distance matrix
distance = np.zeros((num_cities, num_cities))

for i in range(num_cities):
    for j in range(num_cities):
        distance[i, j] = np.linalg.norm(cities[i] - cities[j])

# Heuristic information
heuristic = 1 / (distance + 1e-10)
np.fill_diagonal(heuristic, 0)

# Initialize pheromone
pheromone = np.ones((num_cities, num_cities))

best_route = None
best_distance = float("inf")
best_distance_history = []


# Construct route for an ant
def construct_route(start_city):

    route = [start_city]

    unvisited = set(range(num_cities))
    unvisited.remove(start_city)

    while unvisited:

        current_city = route[-1]
        candidates = list(unvisited)

        weights = [
            (pheromone[current_city, city] ** alpha)
            * (heuristic[current_city, city] ** beta)
            for city in candidates
        ]

        total = sum(weights)

        if total == 0:
            next_city = random.choice(candidates)
        else:
            probabilities = [w / total for w in weights]

            next_city = random.choices(
                candidates,
                weights=probabilities,
                k=1
            )[0]

        route.append(next_city)
        unvisited.remove(next_city)

    # Return to starting city
    route.append(start_city)

    return route


# Calculate route distance
def route_length(route):

    total = 0

    for i in range(len(route) - 1):
        total += distance[route[i], route[i + 1]]

    return total


# -------------------- MAIN ACO LOOP --------------------

for iteration in range(1, num_iterations + 1):

    routes = []
    lengths = []

    # Each ant constructs a route
    for ant in range(num_ants):

        start = random.randrange(num_cities)

        route = construct_route(start)

        length = route_length(route)

        routes.append(route)
        lengths.append(length)

        # Update global best route
        if length < best_distance:

            best_distance = length
            best_route = route.copy()

    # Pheromone evaporation
    pheromone *= (1 - rho)

    # Pheromone deposition
    for route, length in zip(routes, lengths):

        deposit = Q / length

        for i in range(len(route) - 1):

            a = route[i]
            b = route[i + 1]

            pheromone[a, b] += deposit
            pheromone[b, a] += deposit

    # Store best distance for convergence graph
    best_distance_history.append(best_distance)

    # PRINT AFTER EVERY ITERATION
    print(f"Iteration: {iteration} Best Distance: {best_distance:.2f}")


# -------------------- FINAL OUTPUT --------------------

print("Best Route:", best_route)
print(f"Shortest Distance: {best_distance:.2f}")


# -------------------- GRAPH 1: BEST ROUTE --------------------

plt.figure(figsize=(8, 6))

route_coords = cities[best_route]

plt.plot(route_coords[:, 0], route_coords[:, 1], marker="o")

for i, (x, y) in enumerate(cities):

    plt.annotate(
        str(i),
        (x, y),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.title("ACO - Best TSP Route")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.tight_layout()
plt.show()


# -------------------- GRAPH 2: CONVERGENCE --------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, num_iterations + 1),
    best_distance_history,
    marker="."
)

plt.title("ACO Convergence Graph")
plt.xlabel("Iteration")
plt.ylabel("Best Distance")
plt.grid(True)
plt.tight_layout()
plt.show()
```
```
Iteration: 1 Best Distance: 30.28
Iteration: 2 Best Distance: 27.71
Iteration: 3 Best Distance: 27.71
Iteration: 4 Best Distance: 27.71
Iteration: 5 Best Distance: 27.71
Iteration: 6 Best Distance: 27.71
Iteration: 7 Best Distance: 27.71
Iteration: 8 Best Distance: 27.71
Iteration: 9 Best Distance: 27.71
Iteration: 10 Best Distance: 27.71
Iteration: 11 Best Distance: 27.71
Iteration: 12 Best Distance: 27.71
Iteration: 13 Best Distance: 27.71
Iteration: 14 Best Distance: 27.71
Iteration: 15 Best Distance: 27.71
Iteration: 16 Best Distance: 27.71
Iteration: 17 Best Distance: 27.71
Iteration: 18 Best Distance: 27.71
Iteration: 19 Best Distance: 27.71
Iteration: 20 Best Distance: 27.71
Iteration: 21 Best Distance: 27.71
Iteration: 22 Best Distance: 27.71
Iteration: 23 Best Distance: 27.71
Iteration: 24 Best Distance: 27.71
Iteration: 25 Best Distance: 27.71
Iteration: 26 Best Distance: 27.71
Iteration: 27 Best Distance: 27.71
Iteration: 28 Best Distance: 27.71
Iteration: 29 Best Distance: 27.71
Iteration: 30 Best Distance: 27.71
Iteration: 31 Best Distance: 27.71
Iteration: 32 Best Distance: 27.71
Iteration: 33 Best Distance: 27.71
Iteration: 34 Best Distance: 27.71
Iteration: 35 Best Distance: 27.71
Iteration: 36 Best Distance: 27.71
Iteration: 37 Best Distance: 27.71
Iteration: 38 Best Distance: 27.71
Iteration: 39 Best Distance: 27.71
Iteration: 40 Best Distance: 27.71
Iteration: 41 Best Distance: 27.71
Iteration: 42 Best Distance: 27.71
Iteration: 43 Best Distance: 27.71
Iteration: 44 Best Distance: 27.71
Iteration: 45 Best Distance: 27.71
Iteration: 46 Best Distance: 27.71
Iteration: 47 Best Distance: 27.71
Iteration: 48 Best Distance: 27.71
Iteration: 49 Best Distance: 27.71
Iteration: 50 Best Distance: 27.71
Iteration: 51 Best Distance: 27.71
Iteration: 52 Best Distance: 27.71
Iteration: 53 Best Distance: 27.71
Iteration: 54 Best Distance: 27.71
Iteration: 55 Best Distance: 27.71
Iteration: 56 Best Distance: 27.71
Iteration: 57 Best Distance: 27.71
Iteration: 58 Best Distance: 27.71
Iteration: 59 Best Distance: 27.71
Iteration: 60 Best Distance: 27.71
Iteration: 61 Best Distance: 27.71
Iteration: 62 Best Distance: 27.71
Iteration: 63 Best Distance: 27.71
Iteration: 64 Best Distance: 27.71
Iteration: 65 Best Distance: 27.71
Iteration: 66 Best Distance: 27.71
Iteration: 67 Best Distance: 27.71
Iteration: 68 Best Distance: 27.71
Iteration: 69 Best Distance: 27.71
Iteration: 70 Best Distance: 27.71
Iteration: 71 Best Distance: 27.71
Iteration: 72 Best Distance: 27.71
Iteration: 73 Best Distance: 27.71
Iteration: 74 Best Distance: 27.71
Iteration: 75 Best Distance: 27.71
Iteration: 76 Best Distance: 27.71
Iteration: 77 Best Distance: 27.71
Iteration: 78 Best Distance: 27.71
Iteration: 79 Best Distance: 27.71
Iteration: 80 Best Distance: 27.71
Iteration: 81 Best Distance: 27.71
Iteration: 82 Best Distance: 27.71
Iteration: 83 Best Distance: 27.71
Iteration: 84 Best Distance: 27.71
Iteration: 85 Best Distance: 27.71
Iteration: 86 Best Distance: 27.71
Iteration: 87 Best Distance: 27.71
Iteration: 88 Best Distance: 27.71
Iteration: 89 Best Distance: 27.71
Iteration: 90 Best Distance: 27.71
Iteration: 91 Best Distance: 27.71
Iteration: 92 Best Distance: 27.71
Iteration: 93 Best Distance: 27.71
Iteration: 94 Best Distance: 27.71
Iteration: 95 Best Distance: 27.71
Iteration: 96 Best Distance: 27.71
Iteration: 97 Best Distance: 27.71
Iteration: 98 Best Distance: 27.71
Iteration: 99 Best Distance: 27.71
Iteration: 100 Best Distance: 27.71
Best Route: [2, 9, 0, 5, 1, 6, 3, 7, 8, 4, 2]
Shortest Distance: 27.71
```
<img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/ef43252f-8074-4be7-9f8e-71f078326d59" />
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/d43fe264-d924-4646-aebe-e998e64a74db" />


