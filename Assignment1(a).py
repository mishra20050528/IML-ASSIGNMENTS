# ============================================
# QUESTION 1
# Initialize a 1D Array and Convert into
# NumPy and Pandas
# ============================================

import numpy as np
import pandas as pd

# Python List
arr = [10, 20, 30, 40, 50]

# Convert into NumPy Array
numpy_array = np.array(arr)

# Convert into Pandas Series
pandas_series = pd.Series(arr)

print("Original List:")
print(arr)

print("\nNumPy Array:")
print(numpy_array)

print("\nPandas Series:")
print(pandas_series)