from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,test_size=0.3)

# features:
#   sepal length
#   sepal width
#   petal length
#   petal width

model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_hat = model.predict(X_test)

print(confusion_matrix(y_test,y_hat))

with open('model.pkl','wb') as f:
    pickle.dump(model,f)

