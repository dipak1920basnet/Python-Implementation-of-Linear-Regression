from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
coefficients = model.coef_

mse = mean_squared_error(y_test, predictions)
mae = mean_squared_error(y_test, predictions)

print("Coefficient: ", coefficients)
print("Mean Sqaured Error: ", mse)
print("Mean Absolute Error: ", mae)

