import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [20000, 30000, 45000, 49000, 50000]
})

# Split features and target
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Create polynomial features (degree 4)
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

# Fit polynomial regression model
model_pol = LinearRegression()
model_pol.fit(x_poly, y)

# Predict values
y_pred = model_pol.predict(x_poly)

# Calculate metrics
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)

# Print metrics
print("R² Score      :", r2)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)

# Generate smooth curve for plotting
x_grid = np.linspace(min(x), max(x), 100).reshape(-1, 1)
y_grid = model_pol.predict(poly_reg.transform(x_grid))

# Basic plot
plt.scatter(x, y, color='red')
plt.plot(x_grid, y_grid, color='blue')
plt.title('Polynomial Regression (Degree 4)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# x_grid = np.arange(min(x), max(x) + 0.1, 0.1).reshape(-1, 1)
# y_grid_pred = model_pol.predict(poly_reg.transform(x_grid))

# plt.scatter(x, y, color='red', label='Actual Data')
# plt.plot(x_grid, y_grid_pred, color='blue', label='Polynomial Regression Curve')
# plt.title('Polynomial Regression (Degree 4)')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()
