import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from yahoofinancials import YahooFinancials
import datetime
import spacy
from spacy.matcher import PhraseMatcher
import re
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from time import sleep

#real path:
#text_path = '20061020_20131126_bloomberg_news\*\*'
#test path:
text_path = '20061020_20131126_bloomberg_news_test\*\*'

#my tickers
ticker_path = r'ticker_list.csv'

#load spacy nlp core
nlp = spacy.load("en_core_web_sm")

def match_maker(input_ticker_path):
    with open (input_ticker_path) as tl:
        tldf = pd.read_csv(tl, dtype="string")
        ticker_list = tldf['ticker'].to_list()
        #company_list = tldf['names_1'].to_list()
        #extra name variations:
            #company_list2 = tldf['names_2']
            #company_list2 = tldf['names_2'].dropna()
            #company_list2 = company_list2.to_list()
            #company_list3 = tldf['names_3']
            #company_list3 = tldf['names_3'].dropna()
            #company_list3 = company_list3.to_list()
            #putting lists together:
            #company_list = (company_list + company_list2 + company_list3)
    
    #will need to find a way to revert company names to ticker names

#gather names from gazetteer lists and put into spacy nlp thing:    
    ticker_patterns = [nlp(text) for text in ticker_list]
#company_patterns = [nlp(text) for text in company_list]

#prepare matcher and add patters from above:
    matcher = PhraseMatcher(nlp.vocab)
    matcher.add('TICKER', None, *ticker_patterns)
    #matcher.add('COMPANY', None, *company_patterns)
    
    return matcher

def gather_docs(path):
#this gets all my subdirectories into a list
    ls_documents = [] 
    for name in glob.glob(path):
        ls_documents.append(name)

#this goes into the above subdirectory list and opens each file, and puts each 
#batch of text into a list
    ls_text = []
    for document in ls_documents:
        f = open(document,"r")
        ls_text.append(f.read())
        
    #this removes the author name and URL address from each text file
    ls_text_no_authorURL = [re.sub('.*\n.*\n.*\n.*\n.*\.html', '', text) for text in ls_text]
    #removing \n
    ls_clean = [re.sub('\\n', ' ', text) for text in ls_text_no_authorURL]
    
    return ls_clean
    
def gather_dates(path):
    ls_documents = [] 
    for name in glob.glob(path):
        ls_documents.append(name)

    #this is my list of article dates for each text file
    #NOTE: indices:
    #38:48 for test
    ls_dates = [w[38:48] for w in ls_documents]
    #33:43 for real
    #ls_dates = [w[33:43] for w in ls_documents]

    return ls_dates

def get_ids_and_sents(input_text):
    """
    Takes text as input. Outputs a dictionary 
    with tickers/company names,
    the sentence where the name occurred, 
    and the type of name (TICKER or COMPANY).
    Outputs an empty dictionary if it doesn't find anything.
    """
    text = input_text
    doc = nlp(text)
    matches = matcher(doc)
        
    ids = [np.NaN]
    sents = [np.NaN]
    string_ids = [np.NaN]
    
    
    for match_id, start, end in matches:
        nlp.vocab.strings[match_id]  # Get string representation
        span = doc[start:end]  # The matched span of text
        
        ids.append(str(span.text))
        sents.append(str(span.sent))
        string_id = nlp.vocab.strings[match_id]  
        string_ids.append(string_id)
    
    
                               
    try:
        #output lists to a dictionary:
        myDict = {}        
        myDict["ids"] = ids
        myDict["sents"] = sents
        myDict["string_ids"] = string_ids
        
        #add tricks here!
        return myDict
    except:
        pass
    
def make_sents_dates(input_text_list, input_dates_list):
    counts = []
    appended_data = []

    for i in input_text_list:
        myDicts = get_ids_and_sents(i)
        counts.append(len(myDicts['ids']))
        d = pd.DataFrame(myDicts)
        appended_data.append(d)
       
    df = pd.concat(appended_data)

    dates = sum([[date] * count for date, count in zip(input_dates_list, counts)], [])
    df['dates'] = dates

    #removing nan items:
    df = df.dropna()

#dropping weird author sentences that look like "A n d r e w" so that
#it doesn't screw up my ticker:
    try:
        df = df[~df["sents"].str.contains('[A-Z]\s[a-z]\s[a-z]\s', regex=True)] 
    except:
        pass
    
    return df


from func_timeout import func_set_timeout
import func_timeout

@func_set_timeout(20)
def yahoo_price_getter(open_date, ticker):
    """
    Takes as input a date of the article, along with
    the ticker symbol. Returns a dictionary of price data for the next day's price
    minus the open date's price. Used for below function.
    """
    try:
        yahoo_financials = YahooFinancials(ticker)
    
        date_1 = datetime.datetime.strptime(open_date, "%Y-%m-%d") + datetime.timedelta(days=2)
        date_2 = date_1.date()
        close_date = datetime.datetime.strftime(date_2, "%Y-%m-%d")
    
    
    #uses the above function
        prices = yahoo_financials.get_historical_price_data(start_date=open_date, 
                end_date=close_date, time_interval='daily')
    except:
        pass
    
    try:
        open_price = (prices[ticker]['prices'][0]['open'])
        
        sleep(0.25)
        print("Obtained an opening price!")
    except func_timeout.exceptions.FunctionTimedOut:
        pass
    except:
        return np.NaN
    
    
    try:
        close_price = (prices[ticker]['prices'][1]['close'])
        
        sleep(0.25)
        print("Obtained a closing price!")
        price_diff = close_price - open_price
        if price_diff > 0:
            return 1
        if price_diff <= 0:
            return 0
        else:
            return np.NaN
    except func_timeout.exceptions.FunctionTimedOut:
        pass
    except:
        return np.NaN

def make_df(input_text_path):
    docs = gather_docs(input_text_path)
    print("Done getting docs!")
    dates = gather_dates(input_text_path)
    print("Done getting dates!")
    df = make_sents_dates(docs, dates)
    print("Done making dataframe with sentences and dates!")
    print("Working on prices now!")
    df['buy?'] = df.apply(lambda x: yahoo_price_getter(x['dates'], x['ids']), axis=1)
    
    df = df.dropna()
    print("Done adding prices to dataframe!")
    
    return df

#using above functions to spit out dataframe:
matcher = match_maker(ticker_path)

df = make_df(text_path)

#output to csv:
#df.to_csv(r'data_ticker_only_test.csv')


#append to csv:
df.to_csv('data_ticker_only.csv', mode='a', header=False)
print("Done appending to csv!")

"""
#vectorizer
#get sentiment scores via nltk vader
nltk.download('vader_lexicon')    
sia = SentimentIntensityAnalyzer()
df['sentiment_scores'] = df['sents'].apply(lambda sents: sia.polarity_scores(sents))
df['compound'] = df['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])

#get subjectivity via textblob
df['subj_scores'] = df['sents'].apply(lambda sents: TextBlob(sents).sentiment.subjectivity)

##vectorize and send to array so I can attach other features:
tfidf = TfidfVectorizer(stop_words='english')
x1 = tfidf.fit_transform(df['sents'])
x1 = x1.toarray()

#attach features together:
x2 = df['compound']
x3 = df['subj_scores']
X = np.c_[x1, x2, x3]

y = df['buy?']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=333)

print("Done making word vectors and splitting into train/test!")

#Logistic regression:
clf = LogisticRegression(penalty="l1", C=100, solver="liblinear").fit(X_train, y_train)
LRpreds = clf.predict(X_test)

#wrangling the pandas:
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
""" 