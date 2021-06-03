import os
import pickle
import numpy as np
import tensorflow as tf


INPUT1 = os.path.join("..", "Data", "input_output_full.pkl")
INPUT2 = os.path.join("..", "Data", "silence.pkl")

# Unpickle the correctly formatted input output
pickle_off = open(INPUT1,"rb")
train_set, train_labels, test_set, test_labels = pickle.load(pickle_off)

pickle_off = open(INPUT2,"rb")
silence_set, silence_labels = pickle.load(pickle_off)

# Let's count the categories
labels = tf.concat([train_labels, test_labels, silence_labels], axis=0)
labels = labels.numpy()

unique, freq = np.unique(labels, return_counts=True)
# print unique values array
print("Unique Values:", 
      unique)
  
# print frequency array
print("Frequency Values:",
      freq)

# Numpy
silence_set, silence_labels = silence_set.numpy(), silence_labels.numpy()
train_set, train_labels = train_set.numpy(), train_labels.numpy()
test_set, test_labels = test_set.numpy(), test_labels.numpy()
# Sort sets
temp_set = np.vstack((train_set, test_set))
temp_labels = np.hstack((train_labels, test_labels))
neg_set = temp_set[temp_labels == 0]
hs_set = temp_set[temp_labels == 1]
neg_labels = np.zeros(neg_set.shape[0])
hs_labels = np.ones(hs_set.shape[0])
# Balance
silence_set = np.repeat(silence_set, 130, axis=0)
silence_labels = np.repeat(silence_labels, 130, axis=0)
hs_set = np.repeat(hs_set, 8, axis=0)
hs_labels = np.repeat(hs_labels, 8, axis=0)
# Concatenate
temp_set = np.vstack((silence_set, neg_set, hs_set))
temp_labels = np.hstack((silence_labels, neg_labels, hs_labels))
# Shuffle
idx = np.arange(temp_labels.shape[0])
np.random.shuffle(idx)
temp_set = temp_set[idx]
temp_labels = temp_labels[idx]
# Split
test_set = temp_set[:int(train_set.shape[0]/5)]
train_set = temp_set[int(train_set.shape[0]/5):]
test_labels = temp_labels[:int(train_set.shape[0]/5)]
train_labels = temp_labels[int(train_set.shape[0]/5):]
# Pickle

enhanced_dataset = (train_set, train_labels, test_set, test_labels)
PATHTODUMP = os.path.join("..", "Data", "enhanced_dataset.pkl")
with open(PATHTODUMP, 'wb') as f:
  pickle.dump(enhanced_dataset, f)

print("Done.")