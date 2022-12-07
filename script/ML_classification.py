import pandas as pd

# heart disease dataset
data = pd.read_csv(r"D:\GitHub\MIS_584\cleveland.csv")

print(data.shape)
print(list(data.columns))

# missing values
print((data == '?').sum(axis = 0))
data.loc[data['ca'] == '?', 'ca'] = 0
data.loc[data['thal'] == '?', 'thal'] = 3
print((data == '?').sum(axis = 0))

# create training data and labels
X = data.drop(columns = 'num')
y = data['num']


# =============================================================================
# K-Nearest Neighbor Classification
# =============================================================================

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X, y)

knn_pred = knn.predict(X)
knn_score = knn.predict_proba(X)

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# confusion matrix
print("Confusion matrix:")
knn_conf_mat = confusion_matrix(y, knn_pred)
print(knn_conf_mat)
print()

# accuracy
knn_accuracy = accuracy_score(y, knn_pred)
print("Prediction accuracy: {:.4f}".format(knn_accuracy))

# recall
knn_recall = recall_score(y, knn_pred)
print("Prediction recall: {:.4f}".format(knn_recall))

# precision
knn_precision = precision_score(y, knn_pred)
print("Prediction precision: {:.4f}".format(knn_precision))

# F1 score
knn_f1 = f1_score(y, knn_pred)
print("Prediction F1: {:.4f}".format(knn_f1))

# AUC-ROC
knn_auc = roc_auc_score(y, knn_score[:, 1])
print("AUC-ROC: {:.4f}".format(knn_auc))


# =============================================================================
# Naive Bayes
# =============================================================================

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X, y)

gnb_pred = gnb.predict(X)
gnb_score = gnb.predict_proba(X)

# confusion matrix
print("Confusion matrix:")
gnb_conf_mat = confusion_matrix(y, gnb_pred)
print(gnb_conf_mat)
print()

# accuracy
gnb_accuracy = accuracy_score(y, gnb_pred)
print("Prediction accuracy: {:.4f}".format(gnb_accuracy))

# recall
gnb_recall = recall_score(y, gnb_pred)
print("Prediction recall: {:.4f}".format(gnb_recall))

# precision
gnb_precision = precision_score(y, gnb_pred)
print("Prediction precision: {:.4f}".format(gnb_precision))

# F1 score
gnb_f1 = f1_score(y, gnb_pred)
print("Prediction F1: {:.4f}".format(gnb_f1))

# AUC-ROC
gnb_auc = roc_auc_score(y, gnb_score[:, 1])
print("AUC-ROC: {:.4f}".format(gnb_auc))


# =============================================================================
# Logistic Regression 
# =============================================================================

from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression()
log_clf.fit(X, y) # total num of iterations reached limit

log_clf_pred = log_clf.predict(X)
log_clf_score = log_clf.predict_proba(X)

# confusion matrix
print("Confusion matrix:")
log_clf_conf_mat = confusion_matrix(y, log_clf_pred)
print(log_clf_conf_mat)
print()

# accuracy
log_clf_accuracy = accuracy_score(y, log_clf_pred)
print("Prediction accuracy: {:.4f}".format(log_clf_accuracy))

# recall
log_clf_recall = recall_score(y, log_clf_pred)
print("Prediction recall: {:.4f}".format(log_clf_recall))

# precision
log_clf_precision = precision_score(y, log_clf_pred)
print("Prediction precision: {:.4f}".format(log_clf_precision))

# F1 score
log_clf_f1 = f1_score(y, log_clf_pred)
print("Prediction F1: {:.4f}".format(log_clf_f1))

# AUC-ROC
log_clf_auc = roc_auc_score(y, log_clf_score[:, 1])
print("AUC-ROC: {:.4f}".format(log_clf_auc))

# extract logistic regression's features and coefficients
log_clf_coef = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_clf.coef_[0]
    })

print(log_clf_coef)


# =============================================================================
# Support Vector Machine
# =============================================================================

from sklearn.svm import LinearSVC

svm = LinearSVC() # will run for 1000 iterations by default
svm.fit(X, y)

svm_pred = svm.predict(X)

# confusion matrix
print("Confusion matrix:")
svm_conf_mat = confusion_matrix(y, svm_pred)
print(svm_conf_mat)
print()

# accuracy
svm_accuracy = accuracy_score(y, svm_pred)
print("Prediction accuracy: {:.4f}".format(svm_accuracy))

# recall
svm_recall = recall_score(y, svm_pred)
print("Prediction recall: {:.4f}".format(svm_recall))

# precision
svm_precision = precision_score(y, svm_pred)
print("Prediction precision: {:.4f}".format(svm_precision))

# F1 score
svm_f1 = f1_score(y, svm_pred)
print("Prediction F1: {:.4f}".format(svm_f1))


# =============================================================================
# Decision Tree
# =============================================================================

from sklearn.tree import DecisionTreeClassifier, plot_tree

# by default, sklearn expands a decision tree until all leaves are pure
# this leads to overfitting
# specify the max_depth parameter
dt_clf = DecisionTreeClassifier(max_depth = 3) # 
dt_clf.fit(X, y)

dt_clf_pred = dt_clf.predict(X)
dt_clf_score = dt_clf.predict_proba(X)

# confusion matrix
print("Confusion matrix:")
dt_clf_conf_mat = confusion_matrix(y, dt_clf_pred)
print(dt_clf_conf_mat)
print()

# accuracy
dt_clf_accuracy = accuracy_score(y, dt_clf_pred)
print("Prediction accuracy: {:.4f}".format(dt_clf_accuracy))

# recall
dt_clf_recall = recall_score(y, dt_clf_pred)
print("Prediction recall: {:.4f}".format(dt_clf_recall))

# precision
dt_clf_precision = precision_score(y, dt_clf_pred)
print("Prediction precision: {:.4f}".format(dt_clf_precision))

# F1 score
dt_clf_f1 = f1_score(y, dt_clf_pred)
print("Prediction F1: {:.4f}".format(dt_clf_f1))

# AUC-ROC
dt_clf_auc = roc_auc_score(y, dt_clf_score[:, 1])
print("AUC-ROC: {:.4f}".format(dt_clf_auc))

# plot the decision tree
import matplotlib.pyplot as plt

plt.figure(figsize = (24, 24))
plot_tree(dt_clf, feature_names = X.columns)