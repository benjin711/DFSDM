import os
import pickle
import numpy as np
import tensorflow as tf

train_set_ = []
train_labels_ = []
test_set_ = []
test_labels_ = []

for i in range(54):
  pickle_off = open(os.path.join("..", "Data", "input_output_{:02d}.pkl".format(i)),"rb")
  train_set, train_labels, test_set, test_labels = pickle.load(pickle_off)
  if train_set.shape[0] != 0:
    train_set_.append(train_set)
  if train_labels.shape[0] != 0:
    train_labels_.append(train_labels)
  if test_set.shape[0] != 0:
    test_set_.append(test_set)
  if test_labels.shape[0] != 0:
    test_labels_.append(test_labels)

train_set__ = tf.concat(train_set_, 0)
train_labels__ = tf.concat(train_labels_, 0)
test_set__ = tf.concat(test_set_, 0)
test_labels__ = tf.concat(test_labels_, 0)

# Pickle the correctly formatted input output (only quantization normalization missing)
input_output = (train_set__, train_labels__, test_set__, test_labels__)
PATHTODUMP = os.path.join("..", "Data", "input_output_full.pkl")
with open(PATHTODUMP, 'wb') as f:
  pickle.dump(input_output, f)

print("Done.")