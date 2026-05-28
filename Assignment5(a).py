# TITANIC DATASET
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv(r"C:\Users\krish\Downloads\titanic.csv")

# 2. Handle missing values safely
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop unused columns
df = df.drop(columns=['Cabin', 'Ticket', 'Name'])

# 3. Encode categorical variables
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

# 4. Select features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

# 5. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 8. Evaluate
y_pred = knn.predict(X_test)
print("KNN Accuracy (k=5):", accuracy_score(y_test, y_pred))


#Hardcoded KNN
# HARDCODE:
import pandas as pd
import numpy as np

# Load Titanic dataset
df = pd.read_csv(r"C:\Users\krish\Downloads\titanic.csv")

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop irrelevant columns
df = df.drop(columns=['Cabin', 'Ticket', 'Name'])

# Encode categorical variables manually
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Features and target
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
y = df['Survived'].values

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Train-test split (80/20)
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
print("Titanic KNN Accuracy (k=5):", accuracy)