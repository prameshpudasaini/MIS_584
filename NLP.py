reviews = []

# import data
file = open(r"D:\GitHub\MIS_584\text_reviews.txt", encoding = 'utf-8')

for line in file.readlines():
    reviews.append(line.split('\t')[1])
    
# =============================================================================
# Data pre-processing
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

new_reviews = []

for review in reviews:
    # tokenize each review
    review_tokens = nltk.word_tokenize(review)
    
    review_processed = []
    for token in review_tokens:
        # lowercase each token
        token = token.lower()
        
        # remove punctuations
        if token in punctuations:
            continue
        
        # remove stop words
        if token in stop_words:
            continue
        
        # reduce each word to root format
        review_processed.append(ps.stem(token))
    
    review_processed_string = ' '.join(review_processed)
    new_reviews.append(review_processed_string)
    
for i in range(10):
    print(new_reviews[i])


# =============================================================================
# Convert review text into numerical features
# =============================================================================

# extract bag-of-words for each review
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(max_features = 500)
bow_transformer.fit(new_reviews)
reviews_bow = bow_transformer.transform(new_reviews).toarray()
print(reviews_bow)

# extract TF-IDF vectors for each review
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_transformer = TfidfVectorizer(max_features = 500)
tfidf_transformer.fit(new_reviews)
reviews_tfidf = tfidf_transformer.transform(new_reviews).toarray()
print(reviews_tfidf)
