from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from sklearn.utils import shuffle
from statsmodels.stats.contingency_tables import mcnemar

seed = 333

df = pd.read_csv(r"data_ticker_only.csv")
df.dropna()
#too many false TT, so I cut that ticker:
    #others cut: UA: NWS, NWSA, GM, A, I
df = df[df['ids'] != 'TT']
df = df[df['ids'] != 'IT']
#removing a weird unicode character that popped up everywhere '�'
df.replace('�', '', regex=True)
df.drop_duplicates()
df.reset_index(drop=True, inplace=True)
#shuffle data:
df = shuffle(df, random_state=seed)
df.reset_index(drop=True)
print(df)

#nltk.download('vader_lexicon')    
sia = SentimentIntensityAnalyzer()
df['sentiment_scores'] = df['sents'].apply(lambda sents: sia.polarity_scores(sents))
df['compound'] = df['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])

#get subjectivity via textblob
df['subj_scores'] = df['sents'].apply(lambda sents: TextBlob(sents).sentiment.subjectivity)


total_len = len(df)
dev_len = round(len(df)*.8)
test_len = round(len(df)*.9)

"""
splits:
[0:dev_len]
[(dev_len + 1):test_len]
[(test_len + 1):total_len]
"""

x_train_sents = df['sents'][0:dev_len]
x_train_subj = df['subj_scores'][0:dev_len]
x_train_sentiments = df['compound'][0:dev_len]

count_vectorizer = CountVectorizer(ngram_range = (1, 1))
tfidf = TfidfTransformer()

count_matrix_train = count_vectorizer.fit_transform(x_train_sents)
x_train_tfidf = tfidf.fit_transform(count_matrix_train)
x_train_tfidf_array = x_train_tfidf.toarray()

x_train_all = np.c_[x_train_tfidf_array, x_train_subj, x_train_sentiments]

x_test_sents = df['sents'][(dev_len + 1):total_len]
x_test_subj = df['subj_scores'][(dev_len + 1):total_len]
x_test_sentiments = df['compound'][(dev_len + 1):total_len]

count_matrix_test = count_vectorizer.transform(x_test_sents)
x_test_tfidf = tfidf.transform(count_matrix_test)
x_test_tfidf_array = x_test_tfidf.toarray()

x_test_all = np.c_[x_test_tfidf_array, x_test_subj, x_test_sentiments]

y_train = df['buy?'][0:dev_len]
y_test = df['buy?'][(dev_len + 1):total_len]


clf = LogisticRegression(penalty="l1", C=.001, solver="liblinear").fit(x_train_tfidf, y_train)
LRpreds = clf.predict(x_test_tfidf)

predsdf = pd.DataFrame(LRpreds)
predsdf = predsdf.rename(columns={0: 'LRpreds'})

actual = y_test.tolist()
actualdf = pd.DataFrame(actual)

#determine accuracy:
predsdf['actual'] = actualdf
predsdf['correct'] = predsdf['LRpreds'] == (predsdf['actual'])
print(predsdf.head())
print("Finished!")
print("accuracy:", (predsdf['correct'].sum()) / len(predsdf['correct']))

### BETA NONSENSE MODEL:

y1_beta = [1] * (650*5)
y0_beta = [0] * (350*5)
y_beta = y1_beta + y0_beta

np.random.seed(seed)

val = 5000

x1_beta = (np.random.randint(1, 6, size = (val, 1)))
x2_beta = (np.random.randint(1, 6, size = (val, 1)))
x3_beta = (np.random.randint(1, 6, size = (val, 1)))
x4_beta = (np.random.randint(1, 6, size = (val, 1)))
x5_beta = (np.random.randint(1, 6, size = (val, 1)))

X_beta = np.c_[x1_beta, x2_beta, x3_beta, x4_beta, x5_beta]

X_train_beta, X_test_beta, y_train_beta, y_test_beta = train_test_split(
    X_beta, y_beta, test_size=0.2, random_state=seed)

clf = LogisticRegression(penalty="l1", C=.001, solver="liblinear").fit(X_train_beta, y_train_beta)
LRpreds_beta = clf.predict(X_test_beta)

#wrangling the pandas:
predsdf_beta = pd.DataFrame(LRpreds_beta)
predsdf_beta = predsdf_beta.rename(columns={0: 'LRpreds_beta'})

actual_beta = y_test_beta
predsdf_beta['actual_beta'] = pd.DataFrame(actual_beta)

#determine accuracy:
predsdf_beta['correct_beta'] = predsdf_beta['LRpreds_beta'] == predsdf_beta['actual_beta']
print(predsdf_beta.head())
print("Finished!")
print("accuracy_beta:", (predsdf_beta['correct_beta'].sum()) / len(predsdf_beta['correct_beta']))

correct = predsdf['correct'].sum()
incorrect = len(predsdf['correct']) - predsdf['correct'].sum()

correct_beta = predsdf_beta['correct_beta'].sum()
incorrect_beta = len(predsdf_beta['correct_beta']) - predsdf_beta['correct_beta'].sum()

table = [[correct, incorrect], [correct_beta, incorrect_beta]]
#default is binomial distribution
print(mcnemar(table))