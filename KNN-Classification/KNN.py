##  Importing Required Packages



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

##  Loading dataset

iris = load_iris()
x = iris.data[:, :2]   # ONLY 2 features
y = iris.target

## Normalization 

scaler = StandardScaler() 
X_scaled = scaler.fit_transform(x) # only numerical



##  splitting data into train and test sets 

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)



## Train KNN model on the training data

k = 5  # initial number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)



## Make predictions on the test set


y_pred = knn.predict(X_test)


## Evaluate performance (accuracy)


accuracy = accuracy_score(y_test, y_pred)
print(f"K={k}, Test Accuracy: {accuracy:.2f}")


## Plot decision boundaries and classification results

def plot_decision_boundary(X, y, model):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k')
    plt.xlabel("Sepal length (scaled)")
    plt.ylabel("Sepal width (scaled)")
    plt.title("KNN Decision Boundary (2 Features)")
    plt.show()

plot_decision_boundary(X_scaled, y, knn)




## Experiment with different K values to find the relationship between accuracy and the number of neighbors (k)

k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

plt.plot(k_values, accuracies, marker='o')
plt.xlabel("Number of neighbors (k)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k")
plt.grid(True)
plt.show()

## 	The relationship between k and training set size

train_sizes = [30, 50, 70, 100]
k_values = [1, 3, 5, 7, 9]

for size in train_sizes:
    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train, train_size=size, random_state=42
    )

    acc_list = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_sub, y_sub)
        acc_list.append(accuracy_score(y_test, knn.predict(X_test)))

    plt.plot(k_values, acc_list, marker='o', label=f"Train size={size}")

plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k for different training sizes")
plt.legend()
plt.grid(True)
plt.show()

