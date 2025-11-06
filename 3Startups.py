import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
data= pd.read_csv(r"C:\Users\LENOVO\Desktop\Predective_Analysis\50_Startups.csv")
one = OneHotEncoder(sparse_output=False)
data['State'] = one.fit_transform(data[["State"]])
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
x_train,x_test, y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=45)
model_mul = LinearRegression()
model_mul.fit(x_train,y_train)
y_pred = model_mul.predict(x_test)
r2 = r2_score(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mse)
print("R² Score:", r2)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)

import matplotlib.pyplot as plt

# X = R&D, Y = Marketing, Z = Profit (actual)
x = x_test[:, 0]
y = x_test[:, 2]
z = y_test


fig = plt.figure()
ax = fig.add_subplot(projection='3d')  # simple 3D plot
ax.scatter(x, y, z, c='blue')  # actual profit points

ax.set_xlabel('R&D Spend')
ax.set_ylabel('Marketing Spend')
ax.set_zlabel('Profit')
ax.set_title('3D Plot of Profit')

plt.show()



