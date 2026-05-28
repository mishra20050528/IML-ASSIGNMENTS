import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

# Actual and Predicted Values
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Extract Values
TN, FP, FN, TP = cm.ravel()

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Print Results
print("True Positive (TP):", TP)
print("True Negative (TN):", TN)
print("False Positive (FP):", FP)
print("False Negative (FN):", FN)

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.show()


#Hardcoded Confusion Matrix
import numpy as np
import matplotlib.pyplot as plt

# Actual and Predicted Values
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]

# Initialize
TP = 0
TN = 0
FP = 0
FN = 0

# Calculate TP, TN, FP, FN
for i in range(len(y_true)):

    if y_true[i] == 1 and y_pred[i] == 1:
        TP += 1

    elif y_true[i] == 0 and y_pred[i] == 0:
        TN += 1

    elif y_true[i] == 0 and y_pred[i] == 1:
        FP += 1

    elif y_true[i] == 1 and y_pred[i] == 0:
        FN += 1

# Calculate Metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)

precision = TP / (TP + FP)

recall = TP / (TP + FN)

f1_score = (2 * precision * recall) / (precision + recall)

# Print Results
print("True Positive (TP):", TP)
print("True Negative (TN):", TN)
print("False Positive (FP):", FP)
print("False Negative (FN):", FN)

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

# Confusion Matrix
cm = np.array([
    [TN, FP],
    [FN, TP]
])

# Plot Confusion Matrix
plt.imshow(cm, cmap="Blues")

# Labels
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Tick Labels
plt.xticks([0,1], ["0", "1"])
plt.yticks([0,1], ["0", "1"])

# Display Values
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 va="center",
                 color="black")

plt.colorbar()

plt.show()