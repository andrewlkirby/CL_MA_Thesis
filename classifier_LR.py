from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.utils import shuffle

seed = 4

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


df.reset_index(drop=True)
print(df)

#if you need the lexicon:
#nltk.download('vader_lexicon')    
sia = SentimentIntensityAnalyzer()
df['sentiment_scores'] = df['sents'].apply(lambda sents: sia.polarity_scores(sents))
df['compound'] = df['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])

#get subjectivity via textblob
df['subj_scores'] = df['sents'].apply(lambda sents: TextBlob(sents).sentiment.subjectivity)

#partition values
total_len = len(df)
dev_len = round(len(df)*.7)
test_len = round(len(df)*.8)
secret_len = round(len(df)*.9)

#parition and shuffle!
df_train = df[0:dev_len]
df_train = shuffle(df_train, random_state=seed)

df_dev = df[(dev_len + 1):test_len]
df_dev = shuffle(df_dev, random_state=seed)

df_test = df[(test_len + 1):secret_len]
df_test = shuffle(df_test, random_state=seed)
#df_secret is below

"""
splits:
[0:dev_len]
[(dev_len + 1):test_len]
[(test_len + 1):secret_len]
[(secret_len + 1):total_len]
"""
#x_train
x_train_sents = df_train['sents']
x_train_subj = df_train['subj_scores']
x_train_sentiments = df_train['compound']

count_vectorizer = CountVectorizer(ngram_range = (1, 1))
tfidf = TfidfTransformer()

count_matrix_train = count_vectorizer.fit_transform(x_train_sents)
#for wordbags:
#x_train_tfidf = count_matrix_train
x_train_tfidf = tfidf.fit_transform(count_matrix_train)
x_train_tfidf_array = x_train_tfidf.todense()

x_train_tfsenti = np.c_[x_train_tfidf_array, x_train_sentiments]
x_train_tfsubj = np.c_[x_train_tfidf_array, x_train_subj]
x_train_all = np.c_[x_train_tfidf_array, x_train_subj, x_train_sentiments]


#x dev
x_dev_sents = df_dev['sents']
x_dev_subj = df_dev['subj_scores']
x_dev_sentiments = df_dev['compound']

count_matrix_dev = count_vectorizer.transform(x_dev_sents)
#for wordbags:
#x_test_tfidf = count_matrix_test
x_dev_tfidf = tfidf.transform(count_matrix_dev)
x_dev_tfidf_array = x_dev_tfidf.todense()

x_dev_tfsenti = np.c_[x_dev_tfidf_array, x_dev_sentiments]
x_dev_tfsubj = np.c_[x_dev_tfidf_array, x_dev_subj]
x_dev_all = np.c_[x_dev_tfidf_array, x_dev_subj, x_dev_sentiments]

#x test
x_test_sents = df_test['sents']
x_test_subj = df_test['subj_scores']
x_test_sentiments = df_test['compound']

count_matrix_test = count_vectorizer.transform(x_test_sents)
#for wordbags:
#x_test_tfidf = count_matrix_test
x_test_tfidf = tfidf.transform(count_matrix_test)
x_test_tfidf_array = x_test_tfidf.todense()

x_test_tfsenti = np.c_[x_test_tfidf_array, x_test_sentiments]
x_test_tfsubj = np.c_[x_test_tfidf_array, x_test_subj]
x_test_all = np.c_[x_test_tfidf_array, x_test_subj, x_test_sentiments]

#x secret
df_secret = pd.read_csv(r"data_ticker_only_test.csv")
df_secret.dropna()
#too many false TT, so I cut that ticker:
    #others cut: UA: NWS, NWSA, GM, A, I
df_secret = df_secret[df_secret['ids'] != 'TT']
df_secret = df_secret[df_secret['ids'] != 'IT']
#removing a weird unicode character that popped up everywhere '�'
df_secret.replace('�', '', regex=True)
df_secret.drop_duplicates()
df_secret.reset_index(drop=True, inplace=True)
#shuffle data:
df_secret = shuffle(df_secret, random_state=seed)
df_secret.reset_index(drop=True)

sia = SentimentIntensityAnalyzer()
df_secret['sentiment_scores'] = df_secret['sents'].apply(lambda sents: sia.polarity_scores(sents))
df_secret['compound'] = df_secret['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])

#get subjectivity via textblob
df_secret['subj_scores'] = df_secret['sents'].apply(lambda sents: TextBlob(sents).sentiment.subjectivity)

x_secret_sents = df_secret['sents'][0:(total_len - secret_len)]
x_secret_subj = df_secret['subj_scores'][0:(total_len - secret_len - 1)]
x_secret_sentiments = df_secret['compound'][0:(total_len - secret_len - 1)]

count_matrix_secret = count_vectorizer.transform(x_secret_sents)
#for wordbags:
#x_test_tfidf = count_matrix_test
x_secret_tfidf = tfidf.transform(count_matrix_test)
x_secret_tfidf_array = x_secret_tfidf.todense()

x_secret_tfsenti = np.c_[x_secret_tfidf_array, x_secret_sentiments]
x_secret_tfsubj = np.c_[x_secret_tfidf_array, x_secret_subj]
x_secret_all = np.c_[x_secret_tfidf_array, x_secret_subj, x_secret_sentiments]

#y
y_train = df['buy?'][0:dev_len]
y_dev = df['buy?'][(dev_len + 1):test_len]
y_test = df['buy?'][(dev_len + 1):secret_len]
y_secret = df_secret['buy?'][0:(total_len - secret_len)]


#LR TEST
clf = LogisticRegression(penalty="l1", C=.001, solver="liblinear").fit(x_train_tfidf, y_train)
LRpreds = clf.predict(x_test_tfidf)

predsdf = pd.DataFrame(LRpreds)
predsdf = predsdf.rename(columns={0: 'LRpreds'})

actual = y_test.tolist()
actualdf = pd.DataFrame(actual)

#determine test accuracy:
predsdf['actual'] = actualdf
predsdf['correct'] = predsdf['LRpreds'] == (predsdf['actual'])
print("TF-IDF TEST:")
print("number correct:", (predsdf['correct'].sum()))
print("accuracy:", (predsdf['correct'].sum()) / len(predsdf['correct']))

#LR TF-IDF + SENTIMENT
clf = LogisticRegression(penalty="l1", C=.001, solver="liblinear").fit(x_train_tfsenti, y_train)
LRpreds = clf.predict(x_test_tfsenti)

predsdf = pd.DataFrame(LRpreds)
predsdf = predsdf.rename(columns={0: 'LRpreds'})

actual = y_test.tolist()
actualdf = pd.DataFrame(actual)

#determine accuracy:
predsdf['actual'] = actualdf
predsdf['correct'] = predsdf['LRpreds'] == (predsdf['actual'])
print("TF-IDF + SENTIMENT TEST:")
print("number correct:", (predsdf['correct'].sum()))
print("accuracy:", (predsdf['correct'].sum()) / len(predsdf['correct']))

#LR TF-IDF + SUBJ
clf = LogisticRegression(penalty="l1", C=.001, solver="liblinear").fit(x_train_tfsubj, y_train)
LRpreds = clf.predict(x_test_tfsubj)

predsdf = pd.DataFrame(LRpreds)
predsdf = predsdf.rename(columns={0: 'LRpreds'})

actual = y_test.tolist()
actualdf = pd.DataFrame(actual)

#determine accuracy:
predsdf['actual'] = actualdf
predsdf['correct'] = predsdf['LRpreds'] == (predsdf['actual'])
print("TF-IDF + SUBJECTIVITY TEST:")
print("number correct:", (predsdf['correct'].sum()))
print("accuracy:", (predsdf['correct'].sum()) / len(predsdf['correct']))

#LR all
clf_all = LogisticRegression(penalty="l1", C=.001, solver="liblinear").fit(x_train_all, y_train)
LRpreds = clf_all.predict(x_test_all)

predsdf = pd.DataFrame(LRpreds)
predsdf = predsdf.rename(columns={0: 'LRpreds'})

actual = y_test.tolist()
actualdf = pd.DataFrame(actual)

#determine accuracy:
predsdf['actual'] = actualdf
predsdf['correct'] = predsdf['LRpreds'] == (predsdf['actual'])
print("ALL TEST:")
print("number correct:", (predsdf['correct'].sum()))
print("accuracy:", (predsdf['correct'].sum()) / len(predsdf['correct']))


#LR dev
clf = LogisticRegression(penalty="l1", C=.001, solver="liblinear").fit(x_train_tfidf, y_train)
LRpreds = clf.predict(x_dev_tfidf)

predsdf = pd.DataFrame(LRpreds)
predsdf = predsdf.rename(columns={0: 'LRpreds'})

actual = y_dev.tolist()
actualdf = pd.DataFrame(actual)

#determine dev accuracy:
predsdf['actual'] = actualdf
predsdf['correct'] = predsdf['LRpreds'] == (predsdf['actual'])
print("TF-IDF dev:")
print("number correct:", (predsdf['correct'].sum()))
print("accuracy:", (predsdf['correct'].sum()) / len(predsdf['correct']))

#LR TF-IDF + SENTIMENT
clf = LogisticRegression(penalty="l1", C=.001, solver="liblinear").fit(x_train_tfsenti, y_train)
LRpreds = clf.predict(x_dev_tfsenti)

predsdf = pd.DataFrame(LRpreds)
predsdf = predsdf.rename(columns={0: 'LRpreds'})

actual = y_dev.tolist()
actualdf = pd.DataFrame(actual)

#determine accuracy:
predsdf['actual'] = actualdf
predsdf['correct'] = predsdf['LRpreds'] == (predsdf['actual'])
print("TF-IDF + SENTIMENT dev:")
print("number correct:", (predsdf['correct'].sum()))
print("accuracy:", (predsdf['correct'].sum()) / len(predsdf['correct']))

#LR TF-IDF + SUBJ
clf = LogisticRegression(penalty="l1", C=.001, solver="liblinear").fit(x_train_tfsubj, y_train)
LRpreds = clf.predict(x_dev_tfsubj)

predsdf = pd.DataFrame(LRpreds)
predsdf = predsdf.rename(columns={0: 'LRpreds'})

actual = y_dev.tolist()
actualdf = pd.DataFrame(actual)

#determine accuracy:
predsdf['actual'] = actualdf
predsdf['correct'] = predsdf['LRpreds'] == (predsdf['actual'])
print("TF-IDF + SUBJECTIVITY dev:")
print("number correct:", (predsdf['correct'].sum()))
print("accuracy:", (predsdf['correct'].sum()) / len(predsdf['correct']))

#LR all
clf_all = LogisticRegression(penalty="l1", C=.001, solver="liblinear").fit(x_train_all, y_train)
LRpreds = clf_all.predict(x_dev_all)

predsdf = pd.DataFrame(LRpreds)
predsdf = predsdf.rename(columns={0: 'LRpreds'})

actual = y_dev.tolist()
actualdf = pd.DataFrame(actual)

#determine accuracy:
predsdf['actual'] = actualdf
predsdf['correct'] = predsdf['LRpreds'] == (predsdf['actual'])
print("ALL dev:")
print("number correct:", (predsdf['correct'].sum()))
print("accuracy:", (predsdf['correct'].sum()) / len(predsdf['correct']))


#LR secret
clf = LogisticRegression(penalty="l1", C=.001, solver="liblinear").fit(x_train_tfidf, y_train)
LRpreds = clf.predict(x_secret_tfidf)

predsdf = pd.DataFrame(LRpreds)
predsdf = predsdf.rename(columns={0: 'LRpreds'})

actual = y_secret.tolist()
actualdf = pd.DataFrame(actual)

#determine secret accuracy:
predsdf['actual'] = actualdf
predsdf['correct'] = predsdf['LRpreds'] == (predsdf['actual'])
print("TF-IDF secret:")
print("number correct:", (predsdf['correct'].sum()))
print("accuracy:", (predsdf['correct'].sum()) / len(predsdf['correct']))

#LR TF-IDF + SENTIMENT
clf = LogisticRegression(penalty="l1", C=.001, solver="liblinear").fit(x_train_tfsenti, y_train)
LRpreds = clf.predict(x_secret_tfsenti)

predsdf = pd.DataFrame(LRpreds)
predsdf = predsdf.rename(columns={0: 'LRpreds'})

actual = y_secret.tolist()
actualdf = pd.DataFrame(actual)

#determine accuracy:
predsdf['actual'] = actualdf
predsdf['correct'] = predsdf['LRpreds'] == (predsdf['actual'])
print("TF-IDF + SENTIMENT secret:")
print("number correct:", (predsdf['correct'].sum()))
print("accuracy:", (predsdf['correct'].sum()) / len(predsdf['correct']))

#LR TF-IDF + SUBJ
clf = LogisticRegression(penalty="l1", C=.001, solver="liblinear").fit(x_train_tfsubj, y_train)
LRpreds = clf.predict(x_secret_tfsubj)

predsdf = pd.DataFrame(LRpreds)
predsdf = predsdf.rename(columns={0: 'LRpreds'})

actual = y_secret.tolist()
actualdf = pd.DataFrame(actual)

#determine accuracy:
predsdf['actual'] = actualdf
predsdf['correct'] = predsdf['LRpreds'] == (predsdf['actual'])
print("TF-IDF + SUBJECTIVITY secret:")
print("number correct:", (predsdf['correct'].sum()))
print("accuracy:", (predsdf['correct'].sum()) / len(predsdf['correct']))

#LR all
clf_all = LogisticRegression(penalty="l1", C=.001, solver="liblinear").fit(x_train_all, y_train)
LRpreds = clf_all.predict(x_secret_all)

predsdf = pd.DataFrame(LRpreds)
predsdf = predsdf.rename(columns={0: 'LRpreds'})

actual = y_secret.tolist()
actualdf = pd.DataFrame(actual)

#determine accuracy:
predsdf['actual'] = actualdf
predsdf['correct'] = predsdf['LRpreds'] == (predsdf['actual'])
print("ALL secret:")
print("number correct:", (predsdf['correct'].sum()))
print("accuracy:", (predsdf['correct'].sum()) / len(predsdf['correct']))

print("DATES!")
print("DATES TRAIN:")
print(df['dates'][0:dev_len].head())
print("TRAIN LENGTH:")
print(len(df[0:dev_len]))
print("DATES DEV:")
print(df['dates'][(dev_len + 1):test_len].head())
print("DEV LENGTH:")
print(len(df[(dev_len + 1):test_len]))
print("DATES TEST:")
print(df['dates'][(test_len + 1):secret_len].head())
print(df['dates'][(test_len + 1):secret_len].tail())
print("TEST LENGTH:")
print(len(df[(test_len + 1):secret_len]))
print("DATES SECRET")
print("CHECK FILE!")
print("SECRET LENGTH:")
print(len(df_secret[0:(total_len - secret_len)]))