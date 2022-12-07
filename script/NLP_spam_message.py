import pandas as pd

spam_train = pd.read_csv(r"D:\GitHub\MIS_584\data\spam-train.csv")
spam_test = pd.read_csv(r"D:\GitHub\MIS_584\data\spam-test.csv")

print("Number of messages in training data set:", len(spam_train))
print("Number of messages in test data set:", len(spam_test))

# =============================================================================
# Preprocess Text Data
# =============================================================================

import nltk
nltk.download('punkt')

import string
punctuations = string.punctuation

from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

from nltk.stem import PorterStemmer
ps = PorterStemmer()

# iteratively process each message
def process_msg(spam_data):
  msgs = spam_data.iloc[:, 1]
  new_msgs = []

  for msg in msgs:
    msg_tokens = nltk.word_tokenize(msg)

    msg_processed = []
    for token in msg_tokens:
      token = token.lower()
      if token in punctuations:
        continue
      if token in stop_words:
        continue
      msg_processed.append(ps.stem(token))

    msg_processed_string = ' '.join(msg_processed)
    new_msgs.append(msg_processed_string)

  return(new_msgs)

new_train_msgs = process_msg(spam_train)
new_test_msgs = process_msg(spam_test)

# apply TfidfVectorizer function to extract TF-IDF vectors for each message
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_transformer = TfidfVectorizer(max_features=500)

tfidf_transformer.fit(new_train_msgs)
train_features = tfidf_transformer.transform(new_train_msgs).toarray()
test_features = tfidf_transformer.transform(new_test_msgs).toarray()


# =============================================================================
# Predict Spam Messages
# =============================================================================

# Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

log_clf = LogisticRegression()
log_clf.fit(train_features, spam_train['label'])

log_clf_pred = log_clf.predict(test_features)
log_clf_score = log_clf.predict_proba(test_features)

print()
log_clf_f1 = f1_score(spam_test['label'], log_clf_pred)
print("Logistic Regression F1 score: {:.4f}".format(log_clf_f1))

log_clf_auc = roc_auc_score(spam_test['label'], log_clf_score[:, 1])
print("Logistic Regression ROC AUC: {:.4f}".format(log_clf_auc))
print()

# log_clf_coef = pd.DataFrame({
#     'Feature': tfidf_transformer.get_feature_names_out(),
#     'Coefficient': log_clf.coef_[0]})

# print(log_clf_coef.head())

# Naive Bayes

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(train_features, spam_train['label'])

gnb_pred = gnb.predict(test_features)
gnb_score = gnb.predict_proba(test_features)

gnb_f1 = f1_score(spam_test['label'], gnb_pred)
print("Prediction F1: {:.4f}".format(gnb_f1))

gnb_roc_auc = roc_auc_score(spam_test['label'], gnb_score[:, -1])
print("ROC-AUC: {:.4f}".format(gnb_roc_auc))