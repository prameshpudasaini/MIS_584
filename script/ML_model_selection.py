import pandas as pd

data = pd.read_csv(r"D:\GitHub\MIS_584\cleveland.csv")

print(data.shape)
print(list(data.columns))
print()

# missing values
print((data == '?').sum(axis = 0))
data.loc[data['ca'] == '?', 'ca'] = 0
data.loc[data['thal'] == '?', 'thal'] = 3
print((data == '?').sum(axis = 0))
print()

# create training data and labels
X = data.drop(columns = 'num')
y = data['num']

# =============================================================================
# Train-Valid-Test Split for Hyperparameter Tuning
# =============================================================================

from sklearn.model_selection import train_test_split

# split data into test set and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# split training and validation datasets
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2/0.8, random_state = 1)

print("Training set size: {}".format(X_train.shape[0]))
print("Validation set size: {}".format(X_valid.shape[0]))
print("Test set size: {}".format(X_test.shape[0]))

# select optimal K for KNN method
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

k_values = list(range(1, 21, 2))
print("All values of k:", k_values)

knn_train_performance = []
knn_valid_performance = []
knn_test_performance = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    
    # model prediction on all sets
    knn_train_pred = knn.predict(X_train)
    knn_valid_pred = knn.predict(X_valid)
    knn_test_pred = knn.predict(X_test)
    
    # evaluate model performance on all sets
    knn_train_performance.append(accuracy_score(y_train, knn_train_pred))
    knn_valid_performance.append(accuracy_score(y_valid, knn_valid_pred))
    knn_test_performance.append(accuracy_score(y_test, knn_test_pred))
    
all_performance = pd.DataFrame({
    'k': k_values,
    'train_performance': knn_train_performance,
    'valid_performance': knn_valid_performance,
    'test_performance': knn_test_performance
    })

print(all_performance)

import matplotlib.pyplot as plt

plt.plot(all_performance['k'], all_performance['train_performance'],
         linestyle = '-.', color = 'red', label = 'Train')
plt.plot(all_performance['k'], all_performance['valid_performance'],
         linestyle = '-.', color = 'blue', label = 'Valid')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# identify best k that achieves the highest performance on the validation set
max_valid_performance = all_performance['valid_performance'].max()
k_optim = all_performance.loc[all_performance['valid_performance'] == max_valid_performance, 'k'].iloc[0]
print("Optimal value of k:", k_optim)

# get the best knn method
knn_best = KNeighborsClassifier(n_neighbors = k_optim)
knn_best.fit(X_train, y_train)

knn_best_test_pred = knn_best.predict(X_test)
knn_best_test_performance = accuracy_score(y_test, knn_best_test_pred)

print("Final KNN performance on the test set: {:.4f}".format(knn_best_test_performance))


# =============================================================================
# Cross Validation for Hyperparameter Tuning
# =============================================================================

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# split data into test set and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

n_folds = 5
k_values = list(range(1, 21, 2))
tuned_parameters = [{'n_neighbors': k_values}]

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(
    knn,
    tuned_parameters,
    cv = n_folds,
    scoring = ['accuracy', 'f1'],
    refit = 'accuracy' # specify the metric to select the best hyperparameter
    )

knn_cv.fit(X_train, y_train)
print(knn_cv.cv_results_)

print("Average Accuracy Score: ", knn_cv.cv_results_['mean_test_accuracy'])
print("Average F1 Score: ", knn_cv.cv_results_['mean_test_f1'])

all_performance = pd.DataFrame({
    'k': k_values,
    'accuracy': knn_cv.cv_results_['mean_test_accuracy'],
    'f1': knn_cv.cv_results_['mean_test_f1']
    })

print(all_performance)

# identify optimal value of k
max_accuracy = all_performance['accuracy'].max()
k_optim = all_performance.loc[all_performance['accuracy'] == max_accuracy, 'k'].iloc[0]
print("Optimal value of k: ", k_optim)

# GridSearchCV stores the best model; retraining model not required
# get the best estimator
knn_best = knn_cv.best_estimator_
knn_best_test_pred = knn_best.predict(X_test)
knn_best_test_performance = accuracy_score(y_test, knn_best_test_pred)

print("Final KNN performance on the test set: {:.4f}".format(knn_best_test_performance))

# Final KNN performance on the test set using Train-Valid-Test Split: 0.6557
# Final KNN performance on the test set using Cross Validation: 0.7213