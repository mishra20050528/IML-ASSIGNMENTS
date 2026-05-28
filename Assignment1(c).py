# ============================================
# QUESTION 3
# Equation of Line and Error Calculation
# with Graph Plotting
# ============================================

import numpy as np
import matplotlib.pyplot as plt

# Equation of line: y = mx + c
m = 2
c = 3

# Five Coordinates
x_points = np.array([1, 2, 3, 4, 5])
y_actual = np.array([6, 8, 9, 15, 14])

# Predicted Y values
y_predicted = m * x_points + c

# Error Calculation
errors = y_actual - y_predicted

print("X Values:", x_points)
print("Actual Y Values:", y_actual)
print("Predicted Y Values:", y_predicted)
print("Errors:", errors)

# Create Line for Plot
x_line = np.linspace(0, 6, 100)
y_line = m * x_line + c

# Plot Graph
plt.figure(figsize=(8,6))

# Plot Equation Line
plt.plot(x_line, y_line, label='Line: y = 2x + 3')

# Plot Actual Points
plt.scatter(x_points, y_actual, color='red', label='Actual Points')

# Draw Vertical Error Lines
for i in range(len(x_points)):
    plt.vlines(x_points[i],
               y_predicted[i],
               y_actual[i],
               colors='green',
               linestyles='dashed')

# Labels
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Error Visualization on Line Equation")
plt.legend()
plt.grid(True)

# Show Plot
plt.show()