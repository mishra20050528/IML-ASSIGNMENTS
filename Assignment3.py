import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Sample Dataset
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)

y = np.array([1, 4, 9, 16, 25, 36])

# ==========================================
# Linear Regression
# ==========================================

linear_model = LinearRegression()

linear_model.fit(X, y)

y_linear_pred = linear_model.predict(X)

linear_r2 = r2_score(y, y_linear_pred)

# ==========================================
# Polynomial Regression
# ==========================================

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

poly_model = LinearRegression()

poly_model.fit(X_poly, y)

y_poly_pred = poly_model.predict(X_poly)

poly_r2 = r2_score(y, y_poly_pred)

# ==========================================
# Print R² Scores
# ==========================================

print("Linear Regression R² Score :", linear_r2)

print("Polynomial Regression R² Score :", poly_r2)

# ==========================================
# Plotting
# ==========================================

plt.scatter(X, y, color="blue", label="Actual Data")

# Linear Regression Line
plt.plot(X, y_linear_pred,
         color="red",
         label="Linear Regression")

# Polynomial Regression Curve
plt.plot(X, y_poly_pred,
         color="green",
         label="Polynomial Regression")

plt.title("Linear vs Polynomial Regression")

plt.xlabel("X Values")
plt.ylabel("Y Values")

plt.legend()

plt.grid(True)

plt.show()


#Hardcoded Linear and Polynimial Regression
import numpy as np
import matplotlib.pyplot as plt

# Sample Dataset
X = np.array([1, 2, 3, 4, 5, 6])

y = np.array([1, 4, 9, 16, 25, 36])

# ==========================================
# Hardcoded Linear Regression
# ==========================================

mean_x = np.mean(X)
mean_y = np.mean(y)

numerator = np.sum((X - mean_x) * (y - mean_y))

denominator = np.sum((X - mean_x) ** 2)

b1 = numerator / denominator

b0 = mean_y - (b1 * mean_x)

# Predictions
y_linear_pred = b0 + b1 * X

# ==========================================
# Polynomial Regression using polyfit
# ==========================================

coefficients = np.polyfit(X, y, 2)

y_poly_pred = np.polyval(coefficients, X)

# ==========================================
# R² Score Function
# ==========================================

def r2_score(actual, predicted):

    ss_total = np.sum((actual - np.mean(actual)) ** 2)

    ss_residual = np.sum((actual - predicted) ** 2)

    return 1 - (ss_residual / ss_total)

# Calculate R² Scores
linear_r2 = r2_score(y, y_linear_pred)

poly_r2 = r2_score(y, y_poly_pred)

# ==========================================
# Print Results
# ==========================================

print("Linear Regression R² Score :", linear_r2)

print("Polynomial Regression R² Score :", poly_r2)

# ==========================================
# Plotting
# ==========================================

plt.scatter(X, y, color="blue", label="Actual Data")

# Linear Regression
plt.plot(X, y_linear_pred,
         color="red",
         label="Linear Regression")

# Polynomial Regression
plt.plot(X, y_poly_pred,
         color="green",
         label="Polynomial Regression")

plt.title("Linear vs Polynomial Regression")

plt.xlabel("X Values")
plt.ylabel("Y Values")

plt.legend()

plt.grid(True)

plt.show()
