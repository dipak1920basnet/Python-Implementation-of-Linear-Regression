from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_digits()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2)

model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train, y_train)
prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
print(f"Logistic Regression model accuracy: {(accuracy*100):.2f}%")