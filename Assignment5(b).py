# DIABETES DATASET
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Diabetes dataset
df = pd.read_csv(r"C:\Users\krish\Downloads\diabetes_dataset.csv")

# Replace invalid zeros with median values
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    df[col] = df[col].replace(0, df[col].median())

# Features and target
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Accuracy
y_pred = knn.predict(X_test)
print("Diabetes KNN Accuracy (k=5):", accuracy_score(y_test, y_pred))


#Hardcoded KNN
# HARDCODE:
import pandas as pd
import numpy as np

# Load Diabetes dataset
df = pd.read_csv(r"C:\Users\krish\Downloads\diabetes_dataset.csv")

# Replace invalid zeros with median values
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    df[col] = df[col].replace(0, df[col].median())

X = df.drop(columns=['Outcome']).values
y = df['Outcome'].values

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# KNN function
def knn_predict(X_train, y_train, x, k=5):
    distances = np.sqrt(np.sum((X_train - x)**2, axis=1))
    idx = np.argsort(distances)[:k]
    neighbors = y_train[idx]
    return np.bincount(neighbors).argmax()

# Predictions
y_pred = [knn_predict(X_train, y_train, x, k=5) for x in X_test]

# Accuracy
accuracy = np.mean(y_pred == y_test)
print("Diabetes KNN Accuracy (k=5):", accuracy)