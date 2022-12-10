# File with loading, formating and splitting method
import sys
sys.path.insert(0, "/home/adam/Development/Machine_learning/Study/Sklearn/data")
from data import data, accuracy, datasets

# Other imports
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


# Loading data
X_train, y_train, X_test, y_test = data.load_format_split(datasets.wine())


# Our regular model
# Best with mid-size datasets
model = RandomForestClassifier(n_estimators=5)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

acc = accuracy.calculate(predictions, y_test)

# Printing it out
print(f'\nOur Random Forest Classifier model accuracy = {acc}%')



# Tweking our estimators amount to get best accuracy

acc = []
n_estimators = []
for i in range(15):
    n_estimators.append(i+1)

for i in range(15, 150, 20):
    n_estimators.append(i)

print(n_estimators)

for n in n_estimators:
    model = RandomForestClassifier(n_estimators=n)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc.append(float(accuracy.calculate(predictions, y_test)))

print(f'\nOur accuracy with 1-26 estimators:\n{max(acc)}\n')

# if n_estimators >= 26 then our accuracy will always be the same (76.6)
# best accuracy with 3 or 5 estimators

# Showing graph 
plt.plot(n_estimators, acc)
plt.xlabel('estimators, n')
plt.ylabel('accuracy, %')
plt.show()