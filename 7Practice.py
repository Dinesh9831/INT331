import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ----------------- LOAD DATASET -----------------
data = pd.read_csv(r"C:\Users\LENOVO\Desktop\Predective_Analysis\Employee_Performance_Salary.csv")

# ----------------- ONE-HOT ENCODE CATEGORICAL COLUMNS -----------------
categorical_cols = ['Department', 'Education_Level']
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_array = encoder.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))
encoded_df.index = data.index
data_final = pd.concat([data.drop(categorical_cols, axis=1), encoded_df], axis=1)

# ----------------- FEATURES AND TARGET -----------------
X = data_final.drop(['Salary', 'Employee_ID'], axis=1).values
y = data_final['Salary'].values

# ----------------- TRAIN-TEST SPLIT -----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----------------- LINEAR REGRESSION -----------------
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_train_pred_lin = lin_model.predict(X_train)
y_test_pred_lin = lin_model.predict(X_test)

# Plot Linear Regression
plt.scatter(X_train[:,0], y_train, color="blue", label="Train Data")
plt.scatter(X_test[:,0], y_test, color="green", label="Test Data")
sorted_idx = X_train[:,0].argsort()
plt.plot(X_train[sorted_idx,0], y_train_pred_lin[sorted_idx], color="red", label="Linear Fit")
plt.title("Linear Regression")
plt.xlabel("Feature 1")
plt.ylabel("Salary")
plt.legend()
plt.show()

print("Linear Regression R² (test):", r2_score(y_test, y_test_pred_lin))

# ----------------- POLYNOMIAL REGRESSION (DEGREE 1) -----------------
degree = 1
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_train_pred_poly = poly_model.predict(X_train_poly)
y_test_pred_poly = poly_model.predict(X_test_poly)

# Plot Polynomial Regression (Degree 1)
plt.scatter(X_train[:,0], y_train, color="blue", label="Train Data")
plt.scatter(X_test[:,0], y_test, color="green", label="Test Data")
plt.plot(X_train[sorted_idx,0], y_train_pred_poly[sorted_idx], color="red", label=f"Polynomial Fit (deg={degree})")
plt.title(f"Polynomial Regression (Degree {degree})")
plt.xlabel("Feature 1")
plt.ylabel("Salary")
plt.legend()
plt.show()

print(f"Polynomial Regression R² (test, degree={degree}):", r2_score(y_test, y_test_pred_poly))


