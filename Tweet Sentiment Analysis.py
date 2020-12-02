#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


tweets = pd.read_csv(r"C:\Users\User\Downloads\twitter.csv")


# In[3]:


tweets


# In[4]:


tweets.info()


# In[5]:


tweets['tweet']


# In[6]:


tweets


# In[7]:


sns.heatmap(tweets.isnull(), yticklabels = False, cbar = False, cmap = "Blues")


# In[8]:


sns.countplot(tweets["label"])


# In[9]:


tweets["num_of_chars"] = tweets["tweet"].apply(len)


# In[10]:


tweets


# In[11]:


tweets["num_of_chars"].plot(bins = 100, kind = "hist")


# In[12]:


tweets.describe()


# In[13]:


tweets[tweets["num_of_chars"] == 11]["tweet"].iloc[0]


# In[14]:


pos = tweets[tweets["label"] == 0]
neg = tweets[tweets["label"] == 1]


# In[15]:


pos


# In[16]:


neg


# # Wordcloud Plotting

# In[17]:


from wordcloud import WordCloud
#Notes of imrovement: Plot Postive and negative side by side


# In[18]:


sentence = tweets["tweet"].tolist()


# In[19]:


superstring = " ".join(sentence)


# In[20]:


plt.figure(figsize = (20,20))
plt.imshow(WordCloud().generate(superstring))


# # Data Cleaning (Removing Stopwords and Punctuation Marks)

# In[42]:


import string
import nltk
list_stopwords =  stopwords.words("english")
def text_cleaner(text):
    text_no_punc = [char for char in text if char not in string.punctuation]
    text_no_punc_joined = "".join(text_no_punc)
    text_no_punc_joined_clean = [word for word in text_no_punc_joined.split() if word.lower() not in list_stopwords]
    return text_no_punc_joined_clean


# In[43]:


clean_tweets = tweets["tweet"].apply(text_cleaner)


# In[44]:


print(clean_tweets[5])


# In[45]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = text_cleaner)
tweets_countvec = CountVectorizer(analyzer = text_cleaner, dtype = "uint8").fit_transform(tweets["tweet"]).toarray()


# In[46]:


tweets_countvec.shape


# In[52]:


X = tweets_countvec
y = tweets["label"]


# # Naive Bayes Model

# In[55]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[57]:


from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


# # Assessment Via Confusion Matrix

# In[59]:


from sklearn.metrics import classification_report, confusion_matrix


# In[60]:


y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot = True )


# In[61]:


print(classification_report(y_test, y_predict_test))


# In[ ]:




