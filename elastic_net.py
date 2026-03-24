from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error

X,y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2)

model = ElasticNet()
model.fit(X_train)
prediction = model.predict(X_test)
coefficient = model.coef_

mse = mean_squared_error(y_test, prediction)
mae= mean_absolute_error(y_test, prediction)

print("Coefficients: ", coefficient)
print("Mean Square Error: ", mse)
print("Mean absolute Error: ", mae)