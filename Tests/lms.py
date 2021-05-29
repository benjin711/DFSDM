import numpy as np
from scipy.io import wavfile
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

INPUT = "LogMelSpec/test01/test01"
OUTPUT = "LogMelSpec/test01/test02"
SAMPLERATE = 9524


audio = pd.read_csv(INPUT + ".log", header=None, encoding='unicode_escape')
lms = pd.read_csv(OUTPUT + ".log", header=None, encoding='unicode_escape')
audio_fil = []
lms_fil = []    

noisy_items = 0
for item in list(audio[0]):
    try:
        audio_fil.append(np.float(item))
    except ValueError:
        noisy_items += 1

for item in list(lms[0]):
    try:
        lms_fil.append(np.float(item))
    except ValueError:
        noisy_items += 1
        
print("Num noisy items: {}".format(noisy_items))

audio_fil = np.array(audio_fil).astype(np.float32)
lms_fil = np.array(lms_fil).astype(np.float32)

### Calculate LMS using tensorflow
sample_rate = 9524.0
lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 4700.0, 80
frame_length = 1024
num_mfcc = 13

stfts = tf.signal.stft(tf.convert_to_tensor(audio_fil[np.newaxis, ...]), 1024, 1024, 1024)[:,:,:-1]
spectrograms = tf.abs(stfts)
spectrograms = tf.reshape(spectrograms, (spectrograms.shape[0],spectrograms.shape[1],-1))
num_spectrogram_bins = stfts.shape[-1]
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
  num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
  upper_edge_hertz)
mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1) # 512, 80
log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)


# Plotting
fig, (ax) = plt.subplots(nrows=1, ncols=1, sharex=True)
xaxis = np.arange(num_mel_bins)
# Audio
#ax1.plot(xaxis, audio_fil)

# Compare LMS
ax.plot(xaxis, lms_fil, color="blue")
ax.plot(xaxis, log_mel_spectrograms.numpy().flatten(), color="red")


ax.set_xlabel("sample", fontsize=18)
ax.set_ylabel("LMS", fontsize=10)

plt.tight_layout()
plt.show()