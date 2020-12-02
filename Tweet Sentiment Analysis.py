import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

tweets = pd.read_csv(r"C:\Users\User\Downloads\twitter.csv")
tweets.info()
sns.heatmap(tweets.isnull(), yticklabels = False, cbar = False, cmap = "Blues")
sns.countplot(tweets["label"])
tweets["num_of_chars"] = tweets["tweet"].apply(len)
tweets["num_of_chars"].plot(bins = 100, kind = "hist")
tweets.describe()

pos = tweets[tweets["label"] == 0]
neg = tweets[tweets["label"] == 1]
# Wordcloud Plotting
sentence = tweets["tweet"].tolist()
superstring = " ".join(sentence)
plt.figure(figsize = (20,20))
plt.imshow(WordCloud().generate(superstring))

#Data Cleaning (Removing Stopwords and Punctuation Marks)
import string
import nltk
list_stopwords =  stopwords.words("english")
def text_cleaner(text):
    text_no_punc = [char for char in text if char not in string.punctuation]
    text_no_punc_joined = "".join(text_no_punc)
    text_no_punc_joined_clean = [word for word in text_no_punc_joined.split() if word.lower() not in list_stopwords]
    return text_no_punc_joined_clean

clean_tweets = tweets["tweet"].apply(text_cleaner)


vectorizer = CountVectorizer(analyzer = text_cleaner)
tweets_countvec = CountVectorizer(analyzer = text_cleaner, dtype = "uint8").fit_transform(tweets["tweet"]).toarray()
tweets_countvec.shape

X = tweets_countvec
y = tweets["label"]

#Naive Bayes Model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


#Assessment Via Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot = True )
