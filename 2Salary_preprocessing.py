# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# # Load the dataset
# data = pd.read_csv(r"C:\Users\LENOVO\Desktop\Predective_Analysis\Salary_Data.csv", encoding='ISO-8859-1')

# # Splitting the data
# x = data.iloc[:, :-1].values
# y = data.iloc[:, -1].values

# # Create and fit the model
# model = LinearRegression()
# model.fit(x, y)

# # Predict values
# y_pred = model.predict(x)

# # Calculate evaluation metrics
# r2 = r2_score(y, y_pred)
# mae = mean_absolute_error(y, y_pred)
# mse = mean_squared_error(y, y_pred)
# rmse = np.sqrt(mse)

# # Print metrics like before
# print("R_Squared Error:", r2)
# print("Mean Absolute Error:", mae)
# print("Mean Squared Error:", mse)
# print("Root Mean Squared Error:", rmse)
# # Plotting actual vs predicted
# plt.scatter(x, y, color='blue')
# plt.plot(x, y_pred, color='red')
# plt.title("Linear Regression: Salary vs Experience")
# plt.xlabel("Years of Experience")
# plt.ylabel("Salary")
# plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load the dataset
data = pd.read_csv(r"C:\Users\LENOVO\Desktop\Predective_Analysis\Salary_Data.csv", encoding='ISO-8859-1')

# Split features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Train-test split (70% train, 30% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predict on test data
Y_pred = model.predict(X_test)

# Calculate metrics
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

# Print results
print("R_Squared Error:", r2)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Plot the results
plt.scatter(X_test, Y_test, color='blue', label='Actual')
plt.plot(X_test, Y_pred, color='red', label='Predicted')
plt.title('Linear Regression: Salary vs Experience (Test Data)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()



