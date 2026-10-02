import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Load dataset
data = pd.read_csv("students.csv")

# Understand dataset
print(data.head())
print("\nDataset Information:")
print(data.info())

# Features
X = data[["Study_Hours", "Attendance", "Assignments"]]

# Target
y = data["Result"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Create model
model = KNeighborsClassifier(n_neighbors=3)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nActual Results:")
print(y_test.values)

print("\nPredicted Results:")
print(y_pred)

print("\nAccuracy:", accuracy)


# Draw histogram
plt.hist(data["Study_Hours"], bins=5, edgecolor="black")

plt.title("Distribution of Study Hours")
plt.xlabel("Study Hours")
plt.ylabel("Number of Students")

plt.show()