# Complete, corrected pipeline — Linear vs Polynomial (degree 4) on scaled Size
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -------------------------
# 1) Dataset
# -------------------------
data = {
    'Size_sqft': [850, 900, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000],
    'Bedrooms': [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    'Location': ['City', 'Suburb', 'City', 'Suburb', 'City', 'Suburb', 'City', 'Suburb', 'City', 'Suburb'],
    'Age': [5, 10, 2, 8, 15, 20, 10, 3, 12, 1],
    'Price': [150000, 130000, 200000, 180000, 250000, 270000, 300000, 320000, 350000, 400000]
}
df = pd.DataFrame(data)

# -------------------------
# 2) One-hot encode location (keeps dataset consistent; not used for plotting)
# -------------------------
df = pd.get_dummies(df, columns=['Location'], drop_first=True)

# -------------------------
# 3) Scale Size (we'll use Size_scaled for modeling & plotting)
# -------------------------
scaler = StandardScaler()
# fit_transform returns a 2D array; assign flattened values to a new column
df['Size_scaled'] = scaler.fit_transform(df[['Size_sqft']]).ravel()

# -------------------------
# 4) Prepare X (scaled Size) and y
# -------------------------
X = df[['Size_scaled']].values    # shape (n_samples, 1)
y = df['Price'].values

# -------------------------
5) Train-test split (fixed random_state for reproducibility)