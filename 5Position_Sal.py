# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# # Load dataset
# data = pd.read_csv(r"C:\Users\LENOVO\Desktop\Predective_Analysis\Position_Salaries.csv")

# # Use only the 'Level' column for X, and 'Salary' for Y
# x = data.iloc[:, 1:2].values   # Level
# y = data.iloc[:, -1].values    # Salary

# # Create polynomial features (degree 4)
# poly_reg = PolynomialFeatures(degree=4)
# x_poly = poly_reg.fit_transform(x)

# # Fit polynomial regression model
# model_pol = LinearRegression()
# model_pol.fit(x_poly, y)

# # Predict values
# y_pred = model_pol.predict(x_poly)

# # Calculate metrics
# r2 = r2_score(y, y_pred)
# mse = mean_squared_error(y, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y, y_pred)

# # Print metrics
# print("R² Score      :", r2)
# print("Mean Squared Error (MSE):", mse)
# print("Root Mean Squared Error (RMSE):", rmse)
# print("Mean Absolute Error (MAE):", mae)

# # Generate smooth curve for plotting
# x_grid = np.arange(min(x), max(x)+0.1, 0.1).reshape(-1, 1)
# y_grid = model_pol.predict(poly_reg.transform(x_grid))

# # Basic plot
# plt.scatter(x, y, color='red')
# plt.plot(x_grid, y_grid, color='blue')
# plt.title('Polynomial Regression (Degree 4)')
# plt.xlabel('Level')
# plt.ylabel('Salary')
# plt.show()


# # import numpy as np
# # import pandas as pd
# # from sklearn.preprocessing import PolynomialFeatures
# # from sklearn.linear_model import LinearRegression
# # from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error

# # # Load your dataset
# # data = pd.read_csv(r"C:\Users\LENOVO\Desktop\Predective_Analysis\Position_Salaries.csv")
# # x = data.iloc[:, 1:2].values
# # y = data.iloc[:, -1].values

# # # Try polynomial degrees from 1 to 6
# # for degree in range(1, 11):
# #     poly = PolynomialFeatures(degree=degree)
# #     x_poly = poly.fit_transform(x)
# #     model = LinearRegression()
# #     model.fit(x_poly, y)
# #     y_pred = model.predict(x_poly)
    
# #     r2 = r2_score(y, y_pred)
# #     mse = mean_squared_error(y, y_pred)
# #     mae = mean_absolute_error(y, y_pred)
    
# #     print(f"Degree {degree}: R² = {r2:.4f}, MSE = {mse:.2f}, MAE = {mae:.2f}")


# # If R² keeps improving a lot → increase the degree.
# # If R² barely improves but MSE stays low → you’ve found your sweet spot.
# # If R² is 1.0 or too perfect → probably overfitting.


# # How to Actually Decide Which Degree Is Better
# # You don’t just guess — you measure using your performance metrics.
# # Usually you’ll look at:
# # R² (closer to 1 = better fit)
# # RMSE or MAE (lower = better)


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# # Load dataset
# data = pd.read_csv(r"C:\Users\LENOVO\Desktop\Predective_Analysis\Position_Salaries.csv")

# # Use only the 'Level' column for X, and 'Salary' for Y
# x = data.iloc[:, 1:-1].values   # Level
# y = data.iloc[:, -1].values    # Salary

# # Create polynomial features (degree 4)
# poly_reg = PolynomialFeatures(degree=4)
# x_poly = poly_reg.fit_transform(x)

# # Fit polynomial regression model
# model_pol = LinearRegression()
# model_pol.fit(x_poly, y)

# # Predict values
# y_pred = model_pol.predict(x_poly)

# # Calculate metrics
# r2 = r2_score(y, y_pred)
# mse = mean_squared_error(y, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y, y_pred)

# # Print metrics
# print("R2 Score      :", r2)
# print("Mean Squared Error (MSE):", mse)
# print("Root Mean Squared Error (RMSE):", rmse)
# print("Mean Absolute Error (MAE):", mae)

# # Plotting Polynomial Regression
# plt.scatter(x,y)
# plt.plot(x,y_pred, color ="Red")
# plt.show()

# # Plotting Linear Regression
# model_l = LinearRegression()
# model_l.fit(x,y)
# y_pred_l = model_l.predict(x)
# plt.scatter(x,y)
# plt.plot(x,y_pred_l,color= "Red")
# plt.show()

# r2 = r2_score(y,y_pred)
# r2_pol = r2_score(y,y_pred_l)
# print(r2)
# print(r2_pol)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# ----------------- LOAD DATASET -----------------
data = pd.read_csv(r"C:\Users\LENOVO\Desktop\Predective_Analysis\Position_Salaries.csv")

# Use only the 'Level' column for X, and 'Salary' for Y
x = data.iloc[:, 1:-1].values   # Level
y = data.iloc[:, -1].values     # Salary

# ----------------- SPLIT DATA -----------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# ----------------- POLYNOMIAL REGRESSION (DEGREE 4) -----------------
poly_reg = PolynomialFeatures(degree=4)
x_train_poly = poly_reg.fit_transform(x_train)
x_test_poly = poly_reg.transform(x_test)

model_pol = LinearRegression()
model_pol.fit(x_train_poly, y_train)

y_pred_train_poly = model_pol.predict(x_train_poly)
y_pred_test_poly = model_pol.predict(x_test_poly)

# ----------------- METRICS -----------------
r2_poly = r2_score(y_test, y_pred_test_poly)
mse_poly = mean_squared_error(y_test, y_pred_test_poly)
rmse_poly = np.sqrt(mse_poly)
mae_poly = mean_absolute_error(y_test, y_pred_test_poly)

print("---- Polynomial Regression ----")
print("R2 Score      :", r2_poly)
print("Mean Squared Error (MSE):", mse_poly)
print("Root Mean Squared Error (RMSE):", rmse_poly)
print("Mean Absolute Error (MAE):", mae_poly)

# ----------------- LINEAR REGRESSION PLOT -----------------
# model_l = LinearRegression()
# model_l.fit(x_train, y_train)
# y_pred_l = model_l.predict(x_train)

# # Sort x_train for proper line plotting
# sorted_idx_lin = x_train[:, 0].argsort()
# x_train_sorted_lin = x_train[sorted_idx_lin]
# y_pred_sorted_lin = y_pred_l[sorted_idx_lin]

# plt.figure(figsize=(8,5))
# plt.scatter(x_train, y_train, color="blue", label="Train Data")
# plt.scatter(x_test, y_test, color="green", label="Test Data")
# plt.plot(x_train_sorted_lin, y_pred_sorted_lin, color="red", label="Linear Fit")
# plt.title("Linear Regression")
# plt.xlabel("Level")
# plt.ylabel("Salary")
# plt.legend()
# plt.show()


# ---------- Linear Regression Plot (No Sorting) ----------
model_l = LinearRegression()
model_l.fit(x_train, y_train)
y_pred_l = model_l.predict(x_train)

plt.figure(figsize=(8,5))
plt.scatter(x_train, y_train, color="blue", label="Train Data")
plt.scatter(x_test, y_test, color="green", label="Test Data")
plt.plot(x_train, y_pred_l, color="red", label="Linear Fit")  # No sorting
plt.title("Linear Regression (No Sorting)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend()
plt.show()


# ----------------- POLYNOMIAL REGRESSION PLOT (DEGREE 4) -----------------
# Sort x_train for smooth curve
sorted_idx_poly = x_train[:, 0].argsort()
x_train_sorted_poly = x_train[sorted_idx_poly]
y_pred_train_sorted_poly = y_pred_train_poly[sorted_idx_poly]

plt.figure(figsize=(8,5))
plt.scatter(x_train, y_train, color="blue", label="Train Data")
plt.scatter(x_test, y_test, color="green", label="Test Data")
plt.plot(x_train_sorted_poly, y_pred_train_sorted_poly, color="red", label="Polynomial Fit (deg=4)")
plt.title("Polynomial Regression (Degree 4)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend()
plt.show()


 # No sorting

# plt.figure(figsize=(8,5))
# plt.scatter(x_train, y_train, color="blue", label="Train Data")
# plt.scatter(x_test, y_test, color="green", label="Test Data")
# plt.plot(x_train, y_pred_train_poly, color="red", label="Polynomial Fit (deg=4)") 
# plt.title("Polynomial Regression (Degree 4) - No Sorting")
# plt.xlabel("Level")
# plt.ylabel("Salary")
# plt.legend()
# plt.show()


