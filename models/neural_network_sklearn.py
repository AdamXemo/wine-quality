import sys
sys.path.insert(0, "/home/adam/Development/Machine_learning/Study/Sklearn/data")
from data import data, accuracy, datasets
from sklearn.neural_network import MLPClassifier

X_train, y_train, X_test, y_test = data.load_format_split(datasets.wine())
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


model = MLPClassifier(hidden_layer_sizes=(11,11,11), max_iter=400)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(accuracy.calculate(predictions, y_test))