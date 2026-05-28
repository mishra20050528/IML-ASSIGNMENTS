import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv(r'C:\\Users\ASUS\OneDrive\Desktop\titanic.csv')
data["Sex"] = data["Sex"].map({"male": 1, "female": 0})
data = data.drop("Name", axis=1)
data = data.drop("Ticket", axis=1)
data = data.drop("Cabin", axis=1)
data = data.drop("Embarked", axis=1)
data["Age"] = data["Age"].interpolate(method='linear')
data
# Split into X and y
X = data.drop("Survived", axis=1)   # remove target
y = data["Survived"]                # target column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (VERY IMPORTANT for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear SVM model
model = SVC(kernel='linear')

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Linear SVM Accuracy(Titanic Dataset):", accuracy_score(y_test, y_pred))

#Kernal SVM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv(r'C:\\Users\ASUS\OneDrive\Desktop\titanic.csv')
data["Sex"] = data["Sex"].map({"male": 1, "female": 0})
data = data.drop("Name", axis=1)
data = data.drop("Ticket", axis=1)
data = data.drop("Cabin", axis=1)
data = data.drop("Embarked", axis=1)
data["Age"] = data["Age"].interpolate(method='linear')
data
# Split into X and y
X = data.drop("Survived", axis=1)   # remove target
y = data["Survived"]                # target column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (VERY IMPORTANT for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear SVM model
model = SVC(kernel='rbf', C=1 ,gamma='scale')

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Kernal SVM Accuracy(Titanic Dataset):", accuracy_score(y_test, y_pred))