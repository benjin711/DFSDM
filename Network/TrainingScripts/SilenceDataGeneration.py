import numpy as np
from scipy.io import wavfile
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import os

def compute_mfccs(tensor, sample_rate, lower_edge_hertz, upper_edge_hertz, num_mel_bins, frame_length, num_mfcc):
    stfts = tf.signal.stft(tensor, frame_length=frame_length, frame_step=frame_length, fft_length=frame_length) 
    spectrograms = tf.abs(stfts)
    spectrograms = tf.reshape(spectrograms, (spectrograms.shape[0],spectrograms.shape[1],-1))
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
      upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :num_mfcc]
    return tf.reshape(mfccs, (mfccs.shape[0],mfccs.shape[1],mfccs.shape[2],-1))

def kinda_random_walk(N):
  krw = []
  norm = np.random.normal(0, 1, N)
  x = 0
  for i in range(N):
    x = (x + norm[i])*0.99
    krw.append(x)
  return np.array(krw)


INPUT = "../LogData/Silence/silence01"
OUTPUT = "../Data/silence.pkl"
SAMPLERATE = 9524


audio = pd.read_csv(INPUT + ".log", header=None, encoding='unicode_escape')
audio_fil = []

noisy_items = 0
for item in list(audio[0]):
    try:
      if(np.float(item) < 500):
        audio_fil.append(float(item))
    except ValueError:
      noisy_items += 1
        
print("Num noisy items: {}".format(noisy_items))
    
audio_fil = np.array(audio_fil).astype(float)

audio_fil = audio_fil[:-(audio_fil.shape[0]%(93*1024))]
audio_fil = audio_fil.reshape(-1, 93, 1024)
audio_fil = audio_fil[0].reshape(-1, 93, 1024)

wavfile.write("silence.wav", 9524, audio_fil.flatten())

audio_fil = np.repeat(audio_fil, 1000, axis=0)

# Shift each sample 
mu = np.random.normal(0, 50, 1000)

for i in range(1000):
  krw = kinda_random_walk(93*1024)
  krw = krw.reshape(93, 1024)
  audio_fil[i] = audio_fil[i] + mu[i]
  audio_fil[i] = audio_fil[i] + krw
  if i < 10:
    wavfile.write("silence{}.wav".format(i), 9524, audio_fil[-1].flatten())


# Sound signal
count = range(1, 93*1024 + 1)

# Visualize 
# plt.plot(count, audio_fil[-1].flatten(), 'r--')
# plt.legend(['Amplitude'])
# plt.xlabel('Count')
# plt.ylabel('Amplitude')
# plt.show()

# wavfile.write("silence.wav", 9524, audio_fil[-1].flatten())


# Pickle the correctly formatted input output (only quantization normalization missing)
silence_data = (audio_fil, 2*np.ones(1000))
PATHTODUMP = os.path.join("..", "Data", "silence.pkl")
with open(PATHTODUMP, 'wb') as f:
  pickle.dump(silence_data, f)


print("Done.")