import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
data = pd.read_csv(r"C:\Users\LENOVO\Desktop\Predective_Analysis\RegressionDataSet.csv", encoding = 'ISO-8859-1')
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# Train model
model = LinearRegression()
model.fit(x, y)

# Predict
y_pred = model.predict(x)

# Calculate metrics
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
# Print results
print("R_Squared Error:", r2)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Plot the regression line
plt.scatter(x, y, color='blue', label='Actual data')
plt.plot(x, y_pred, color='red', label='Predicted line')
plt.title('Linear Regression')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.show()

