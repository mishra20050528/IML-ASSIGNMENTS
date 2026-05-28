# ============================================
# QUESTION 2
# Initialize a 2D Array and Convert into
# 1D Array using NumPy and Pandas
# ============================================

import numpy as np
import pandas as pd

# 2D Array
array_2d = [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]

# Convert into NumPy Array
numpy_2d = np.array(array_2d)

# Convert 2D Array into 1D Array
numpy_1d = numpy_2d.flatten()

# Convert into Pandas DataFrame
df = pd.DataFrame(array_2d)

# Convert DataFrame into 1D Array
pandas_1d = df.values.flatten()

print("Original 2D Array:")
print(numpy_2d)

print("\nNumPy 1D Array:")
print(numpy_1d)

print("\nPandas 1D Array:")
print(pandas_1d)