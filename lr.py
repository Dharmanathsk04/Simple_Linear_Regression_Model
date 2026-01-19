import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([1,2,3,4,5]).reshape(-1,1)

y = np.array([2,4,6,8,10])

model = LinearRegression()

model.fit(x,y)

y_pred = model.predict(x)

print("slope(m):",model.coef_)

print("intercept(c): ",model.intercept_)


print("prediction for  x = 6",model.predict([[6]]))

plt.scatter(x,y)

plt.plot(x,y_pred,color = 'red')

plt.xlabel("X")

plt.ylabel("Y")

plt.title("simple Linear Regression")

plt.show()

