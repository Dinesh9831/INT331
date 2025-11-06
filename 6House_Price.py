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
# 5) Train-test split (fixed random_state for reproducibility)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -------------------------
# 6) Linear Regression (on Size_scaled)
# -------------------------
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

# -------------------------
# 7) Polynomial Regression (degree=4) — pipeline of transform then linear fit
# -------------------------
poly = PolynomialFeatures(degree=4, include_bias=True)
X_train_poly = poly.fit_transform(X_train)   # creates [1, x, x^2, x^3, x^4]
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
X_test_poly = poly.transform(X_test)
y_pred_poly = poly_reg.predict(X_test_poly)

# -------------------------
# 8) Metrics for both models
# -------------------------
def print_metrics(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"{name} -> R2: {r2:.4f} | MSE: {mse:.2f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

print_metrics("Linear Regression", y_test, y_pred_lin)
print_metrics("Polynomial Regression (deg 4)", y_test, y_pred_poly)

# -------------------------
# 9) Plotting: separate plots — scatter of test points + model curve
# -------------------------
# Create a smooth x-grid across the scaled Size range for plotting curves
X_plot = np.linspace(df['Size_scaled'].min(), df['Size_scaled'].max(), 200).reshape(-1, 1)

# Predictions on the grid
y_plot_lin = lin_reg.predict(X_plot)
y_plot_poly = poly_reg.predict(poly.transform(X_plot))

# Plot 1: Linear
plt.figure(figsize=(7, 4))
plt.scatter(X_test, y_test, color='red', label='Test points')
plt.plot(X_plot, y_plot_lin, color='green', linewidth=2, label='Linear fit')
plt.title("Linear Regression")
plt.xlabel("Size (scaled)")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Polynomial
plt.figure(figsize=(7, 4))
plt.scatter(X_test, y_test, color='red', label='Test points')
plt.plot(X_plot, y_plot_poly, color='blue', linewidth=2, label='Polynomial fit (deg 4)')
plt.title("Polynomial Regression (Degree 4)")
plt.xlabel("Size (scaled)")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# # -------------------------
# # Dataset
# # -------------------------
# data = {
#     'Size_sqft': [850, 900, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000],
#     'Bedrooms': [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
#     'Location': ['City', 'Suburb', 'City', 'Suburb', 'City', 'Suburb', 'City', 'Suburb', 'City', 'Suburb'],
#     'Age': [5, 10, 2, 8, 15, 20, 10, 3, 12, 1],
#     'Price': [150000, 130000, 200000, 180000, 250000, 270000, 300000, 320000, 350000, 400000]
# }
# df = pd.DataFrame(data)

# # -------------------------
# # One-hot encode Location
# # -------------------------
# df = pd.get_dummies(df, columns=['Location'], drop_first=True)

# # -------------------------
# # Scale numeric features
# # -------------------------
# scaler = StandardScaler()
# numeric_cols = ['Size_sqft', 'Bedrooms', 'Age']
# df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# # -------------------------
# # Features and target
# # -------------------------
# X = df[numeric_cols].values
# y = df['Price'].values

# # -------------------------
# # Train-test split
# # -------------------------
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # -------------------------
# # Linear regression
# # -------------------------
# lin_model = LinearRegression()
# lin_model.fit(X_train, y_train)
# y_pred_lin = lin_model.predict(X_test)

# # Linear metrics
# print("Linear Regression Metrics:")
# print("R2:", r2_score(y_test, y_pred_lin))
# print("MSE:", mean_squared_error(y_test, y_pred_lin))
# print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lin)))
# print("MAE:", mean_absolute_error(y_test, y_pred_lin))
# print()

# # -------------------------
# # Polynomial regression (degree 4)
# # -------------------------
# poly = PolynomialFeatures(degree=5)
# X_train_poly = poly.fit_transform(X_train)
# X_test_poly = poly.transform(X_test)

# poly_model = LinearRegression()
# poly_model.fit(X_train_poly, y_train)
# y_pred_poly = poly_model.predict(X_test_poly)

# # Polynomial metrics
# print("Polynomial Regression Metrics (deg 4):")
# print("R2:", r2_score(y_test, y_pred_poly))
# print("MSE:", mean_squared_error(y_test, y_pred_poly))
# print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_poly)))
# print("MAE:", mean_absolute_error(y_test, y_pred_poly))
# print()

# # -------------------------
# # Plots (vs Size only)
# # -------------------------
# X_plot = np.linspace(df['Size_sqft'].min(), df['Size_sqft'].max(), 200).reshape(-1, 1)
# mean_bedrooms = X_train[:,1].mean()
# mean_age = X_train[:,2].mean()
# X_plot_full = np.hstack([X_plot, np.full_like(X_plot, mean_bedrooms), np.full_like(X_plot, mean_age)])

# # Predictions for plots
# y_plot_lin = lin_model.predict(X_plot_full)
# y_plot_poly = poly_model.predict(poly.transform(X_plot_full))

# # Plot linear
# plt.scatter(X_train[:,0], y_train, color='blue')
# plt.scatter(X_test[:,0], y_test, color='green')
# plt.plot(X_plot, y_plot_lin, color='red')
# plt.title("Linear Regression")
# plt.xlabel("Size (scaled)")
# plt.ylabel("Price")
# plt.show()

# # Plot polynomial
# plt.scatter(X_train[:,0], y_train, color='blue')
# plt.scatter(X_test[:,0], y_test, color='green')
# plt.plot(X_plot, y_plot_poly, color='red')
# plt.title("Polynomial Regression (Degree 4)")
# plt.xlabel("Size (scaled)")
# plt.ylabel("Price")
# plt.show()







