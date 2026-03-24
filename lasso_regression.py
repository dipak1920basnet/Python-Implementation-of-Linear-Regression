from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, mean_squared_error
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
# Handle categorical data: 

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42, test_size=0.2)
model = Lasso(alpha=0.1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test,predictions)
coeff = model.coef_
print("Mean Square error: ", mse)
print("Coefficents: ", coeff)