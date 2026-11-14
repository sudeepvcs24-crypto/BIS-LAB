# BIS-LAB
RESEARCH PAPER: https://ieeexplore.ieee.org/document/585892
```
PROBLEM STATEMENT:

Feature Selection Using Genetic Algorithm for Breast Cancer Diagnosis involves identifying the most relevant features from the Breast Cancer Wisconsin Diagnostic Dataset to improve classification performance.

Given a dataset containing patient information and multiple features, the objective is to use a Genetic Algorithm (GA) with a Support Vector Machine (SVM) classifier to determine an optimal subset of features that:
1. Selects the most relevant features for breast cancer classification.
2. Removes unnecessary and redundant features.
3. Maximizes classification accuracy using the selected features.
4. Minimizes the number of features while maintaining good prediction accuracy.
5. Compares the classification accuracy using all features and the selected features.
6. Improves the efficiency of the classification process and reduces computational complexity.
```

CODE IMPLEMENTATION


```PY
import random
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

TOTAL_FEATURES = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25,
    random_state=42, stratify=y_train
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

POP_SIZE = 24
GENERATIONS = 30
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.8
PENALTY = 0.01

def fitness(chromosome):

    selected = np.where(chromosome == 1)[0]

    if len(selected) == 0:
        return 0

    model = SVC(kernel="linear")
    model.fit(X_train[:, selected], y_train)

    predictions = model.predict(X_val[:, selected])
    accuracy = accuracy_score(y_val, predictions)

    score = accuracy - PENALTY * (
        len(selected) / TOTAL_FEATURES
    )

    return score


population = [
    np.random.randint(0, 2, TOTAL_FEATURES)
    for _ in range(POP_SIZE)
]

for chromosome in population:
    if np.sum(chromosome) == 0:
        chromosome[random.randrange(TOTAL_FEATURES)] = 1

best_chromosome = None
best_fitness = -1
history = {}

for generation in range(GENERATIONS + 1):

    scores = [fitness(ch) for ch in population]

    index = np.argmax(scores)

    if scores[index] > best_fitness:
        best_fitness = scores[index]
        best_chromosome = population[index].copy()

    if generation in [0, 10, 20, 30]:
        history[generation] = best_fitness

    if generation == GENERATIONS:
        break

    new_population = [best_chromosome.copy()]

    while len(new_population) < POP_SIZE:

        def select_parent():
            ids = random.sample(range(POP_SIZE), 3)
            winner = max(ids, key=lambda i: scores[i])
            return population[winner].copy()

        parent1 = select_parent()
        parent2 = select_parent()

        child1 = parent1.copy()
        child2 = parent2.copy()

        if random.random() < CROSSOVER_RATE:
            point = random.randint(1, TOTAL_FEATURES - 1)

            child1 = np.concatenate(
                [parent1[:point], parent2[point:]]
            )

            child2 = np.concatenate(
                [parent2[:point], parent1[point:]]
            )

        for child in [child1, child2]:
            for i in range(TOTAL_FEATURES):
                if random.random() < MUTATION_RATE:
                    child[i] = 1 - child[i]

            if np.sum(child) == 0:
                child[random.randrange(TOTAL_FEATURES)] = 1

            new_population.append(child)

            if len(new_population) >= POP_SIZE:
                break

    population = new_population[:POP_SIZE]


selected_indices = np.where(best_chromosome == 1)[0]
selected_features = feature_names[selected_indices]

all_model = SVC(kernel="linear")
all_model.fit(X_train, y_train)
all_predictions = all_model.predict(X_test)
all_accuracy = accuracy_score(y_test, all_predictions)

selected_model = SVC(kernel="linear")
selected_model.fit(X_train[:, selected_indices], y_train)
selected_predictions = selected_model.predict(
    X_test[:, selected_indices]
)
selected_accuracy = accuracy_score(y_test, selected_predictions)

print("\n========== INPUT ==========")
print("Dataset: Breast Cancer Dataset")
print("Total Features:", TOTAL_FEATURES)

print("\n========== ITERATIONS ==========")

for iteration, score in history.items():
    print(
        f"Iteration {iteration}: "
        f"Best Fitness = {score:.4f}"
    )

print("\n========== FINAL RESULT ==========")

print("Selected Features:")
for feature in selected_features:
    print("-", feature)

print("\nTotal Features:", TOTAL_FEATURES)
print("Selected Features:", len(selected_indices))
print("Removed Features:",
      TOTAL_FEATURES - len(selected_indices))
print("Best Fitness:", round(best_fitness, 4))

print("\n========== ACCURACY COMPARISON ==========")

print("All Features Accuracy:",
      round(all_accuracy * 100, 2), "%")

print("Selected Features Accuracy:",
      round(selected_accuracy * 100, 2), "%")

plt.figure(figsize=(8, 5))

iterations = list(history.keys())
fitness_values = list(history.values())

plt.plot(iterations, fitness_values,
         marker='o', color='blue')

plt.title("Genetic Algorithm - Fitness Improvement")
plt.xlabel("Iterations")
plt.ylabel("Best Fitness")
plt.grid(True)

plt.show()

plt.figure(figsize=(7, 5))

labels = ["All Features", "Selected Features"]

accuracies = [
    all_accuracy * 100,
    selected_accuracy * 100
]

plt.bar(labels, accuracies,
        color=['blue', 'green'])

plt.title("Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid(axis='y')

plt.show()

```
```
========== INPUT ==========
Dataset: Breast Cancer Dataset
Total Features: 30

========== ITERATIONS ==========
Iteration 0: Best Fitness = 0.9771
Iteration 10: Best Fitness = 0.9791
Iteration 20: Best Fitness = 0.9791
Iteration 30: Best Fitness = 0.9798

========== FINAL RESULT ==========
Selected Features:
- mean concavity
- mean symmetry
- smoothness error
- compactness error
- worst radius
- worst texture
- worst compactness
- worst concave points

Total Features: 30
Selected Features: 8
Removed Features: 22
Best Fitness: 0.9798

========== ACCURACY COMPARISON ==========
All Features Accuracy: 96.49 %
Selected Features Accuracy: 95.61 %
```
<img width="717" height="470" alt="ga1" src="https://github.com/user-attachments/assets/978ed9a6-7a3c-463a-94e8-bc8fcfab5740" />
<img width="618" height="451" alt="ga2" src="https://github.com/user-attachments/assets/921c3b74-2cb2-4e3b-bb9b-dac214f82093" />




