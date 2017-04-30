
# coding: utf-8

# In[1]:


# coding: utf-8

# In[6]:

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import pandas as pd
import re
from tqdm import tqdm
import nltk
import pdb

s = pd.read_csv("yelp_academic_dataset_review_train.csv")

all_words = []
regex = re.compile('[^a-zA-Z]')
for index, item in tqdm(s.iterrows()):
    new_words = item['text'].split();
    new_words = [regex.sub('', w).lower() for w in new_words]
    all_words += new_words

words = all_words

print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 30000

def build_dataset(words, vocabulary_size):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])


# In[ ]:




# In[ ]:




# In[ ]:




# In[2]:




# In[5]:

dictionary["UNK"]


# In[ ]:




# In[ ]:




# In[2]:

import numpy as np

vecs=[]
for index, item in tqdm(s.iterrows()):
    new_words = item['text'].split();
    new_words = [regex.sub('', w).lower() for w in new_words]
    if (len(new_words) > 128):
        new_words = new_words[0:128]
    else:
        new_words += ["" for _ in range(128 - len(new_words))]
    new_words = [dictionary[w] if w in dictionary.keys() else 0 for w in new_words]
    new_words_vec = np.array(new_words, dtype=np.int32).reshape(1,128)
    vecs += [new_words_vec]


op = np.vstack(vecs)
del vecs



# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""

  input_one_hot = tf.one_hot(indices=tf.cast(features, tf.int32), depth=30000)
  input_one_hot = tf.reshape(input_one_hot, [-1,128,1,30000])
  conv1a = tf.layers.conv2d(
      inputs=input_one_hot,
      filters=1000,
      kernel_size=[3,1],
      padding="same",
      activation=tf.nn.relu)
  conv1b = tf.layers.conv2d(
      inputs=input_one_hot,
      filters=1000,
      kernel_size=[2,1],
      padding="same",
      activation=tf.nn.relu)
  conv1 = tf.concat([conv1a, conv1b], 1)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 1], strides=2)
  pool1_flat = tf.reshape(pool1, [-1, 128*1*1000])
  dense = tf.layers.dense(inputs=pool1_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
  ratings = tf.layers.dense(inputs=dropout, units=1, activation=tf.nn.relu6)


  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    labels = tf.reshape(labels, [-1, 1])
    loss = tf.losses.mean_squared_error(labels=labels, predictions=ratings)


  def f(lr, gs):
    return tf.train.exponential_decay(lr, gs, 100, 0.85)
  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate = 0.001,
        learning_rate_decay_fn=f,
        optimizer="SGD")

  # Generate Predictions
  predictions = {
      "ratings": ratings
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_argv):

  train_data = op[0:80000,]
  train_labels = np.array(s["stars"][0:80000])
  eval_data = op[80000:100000,]
  eval_labels = np.array(s["stars"][80000:100000])

  # Create the Estimator
  mnist_classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/ahahahahahaha_convnet_model")

  # Train the model
  mnist_classifier.fit(
      x=train_data,
      y=train_labels,
      batch_size=100,
      steps=20000)

  # Configure the accuracy metric for evaluation
  metrics = {
      "rmse":
          learn.MetricSpec(
              metric_fn=tf.metrics.mean_squared_error, prediction_key="ratings"),
  }

  # Evaluate the model and print results
  eval_results = mnist_classifier.evaluate(
      x=eval_data, y=eval_labels, metrics=metrics)
  print(eval_results)

tf.app.run()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:
