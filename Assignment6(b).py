import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("C:\\Users\ASUS\OneDrive\Desktop\Social_Network_Ads.csv")
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
# Split into X and y
X = data.drop("Purchased", axis=1)   # remove target
y = data["Purchased"]                # target column

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
print("Linear SVM Accuracy(Social Network Dataset):", accuracy_score(y_test, y_pred))

#Kernal SVM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("C:\\Users\ASUS\OneDrive\Desktop\Social_Network_Ads.csv")
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
# Split into X and y
X = data.drop("Purchased", axis=1)
y = data["Purchased"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Kernel SVM (RBF)
model = SVC(kernel='rbf', C=1.0, gamma='scale')

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Kernel SVM Accuracy(Social Network Dataset):", accuracy_score(y_test, y_pred))