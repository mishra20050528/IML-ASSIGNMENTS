import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# ==========================================
# Create Dataset
# ==========================================

X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_classes=2,
    random_state=42
)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Logistic Regression Model
# ==========================================

model = LogisticRegression()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Probability Predictions
y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================
# Evaluation Metrics
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Accuracy :", accuracy)

print("Precision :", precision)

print("Recall :", recall)

print("F1 Score :", f1)

print("\nConfusion Matrix :\n")

print(cm)

# ==========================================
# Plot Confusion Matrix
# ==========================================

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()

# ==========================================
# ROC Curve
# ==========================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))

plt.plot(fpr, tpr,
         label=f"AUC = {roc_auc:.2f}")

plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.show()


#Hardcoded Logistic Regression
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ==========================================
# Create Dataset
# ==========================================

X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_classes=2,
    random_state=42
)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Logistic Regression Model
# ==========================================

model = LogisticRegression()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# ==========================================
# Calculate TP, TN, FP, FN
# ==========================================

TP = 0
TN = 0
FP = 0
FN = 0

for i in range(len(y_test)):

    if y_test[i] == 1 and y_pred[i] == 1:
        TP += 1

    elif y_test[i] == 0 and y_pred[i] == 0:
        TN += 1

    elif y_test[i] == 0 and y_pred[i] == 1:
        FP += 1

    elif y_test[i] == 1 and y_pred[i] == 0:
        FN += 1

# ==========================================
# Metrics
# ==========================================

accuracy = (TP + TN) / (TP + TN + FP + FN)

precision = TP / (TP + FP)

recall = TP / (TP + FN)

f1_score = (2 * precision * recall) / (precision + recall)

# ==========================================
# Print Results
# ==========================================

print("True Positive :", TP)

print("True Negative :", TN)

print("False Positive :", FP)

print("False Negative :", FN)

print("\nAccuracy :", accuracy)

print("Precision :", precision)

print("Recall :", recall)

print("F1 Score :", f1_score)

# ==========================================
# Confusion Matrix Plot
# ==========================================

cm = np.array([
    [TN, FP],
    [FN, TP]
])

plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.xticks([0,1], ["0", "1"])

plt.yticks([0,1], ["0", "1"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 va="center",
                 color="black")

plt.colorbar()

plt.show()

# ==========================================
# ROC Curve (Manual)
# ==========================================

y_prob = model.predict_proba(X_test)[:,1]

thresholds = np.linspace(0, 1, 100)

tpr_list = []
fpr_list = []

for threshold in thresholds:

    y_pred_threshold = (y_prob >= threshold).astype(int)

    TP = TN = FP = FN = 0

    for i in range(len(y_test)):

        if y_test[i] == 1 and y_pred_threshold[i] == 1:
            TP += 1

        elif y_test[i] == 0 and y_pred_threshold[i] == 0:
            TN += 1

        elif y_test[i] == 0 and y_pred_threshold[i] == 1:
            FP += 1

        elif y_test[i] == 1 and y_pred_threshold[i] == 0:
            FN += 1

    TPR = TP / (TP + FN)

    FPR = FP / (FP + TN)

    tpr_list.append(TPR)

    fpr_list.append(FPR)

# Plot ROC Curve
plt.figure(figsize=(8,6))

plt.plot(fpr_list, tpr_list,
         label="ROC Curve")

plt.plot([0,1], [0,1],
         linestyle="--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.show()