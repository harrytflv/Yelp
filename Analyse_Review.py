
# coding: utf-8

# In[2]:

import pandas as pd
import nltk
from tqdm import tqdm


# In[3]:

s = pd.read_csv("yelp_academic_dataset_review_train.csv")
s = s[['text', 'stars']]


# In[4]:

all_words = []
for index, item in tqdm(s.iterrows()):
    new_words = nltk.word_tokenize(item['text']);
    all_words += new_words


# In[5]:

all_words_freq = nltk.FreqDist(all_words)


# In[ ]:




# In[6]:

freq_words = list(all_words_freq.keys())[:2000]


# In[ ]:




# In[7]:

def document_features(document):
    document_words = set(document)
    features = {}
    for word in freq_words:
        features['contains(%s)' % word] = (word in document_words)
    return features


# In[ ]:




# In[8]:

featureset = [(document_features(nltk.word_tokenize(item['text'])), item['stars'] > 4) for index, item in tqdm(s.iterrows())]


# In[ ]:

trainset, testset = featureset[20000:], featureset[:20000]


# In[ ]:

classifier = nltk.NaiveBayesClassifier.train(trainset)


# In[ ]:

print(nltk.classify.accuracy(classifier, testset))