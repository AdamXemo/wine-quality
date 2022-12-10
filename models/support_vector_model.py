# File with loading, formating and splitting method
import sys
sys.path.insert(0, "/home/adam/Development/Machine_learning/Study/Sklearn/data")
from data import data, accuracy, datasets
from sklearn import svm

X_train, y_train, X_test, y_test = data.load_format_split(datasets.wine())

model = svm.SVC()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

acc = accuracy.calculate(predictions, y_test)
print(acc)
