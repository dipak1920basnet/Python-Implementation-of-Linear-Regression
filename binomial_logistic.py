from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X,y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.2)

clf = LogisticRegression(max_iter=1000, random_state=0)
clf.fit(X_train,y_train)
prediction = clf.predict(X_test)
acc = accuracy_score(y_test, prediction)*100

print(f"Logistic Regression model accuracy: {acc:.2f}%")
