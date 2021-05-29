import numpy as np
from scipy.io import wavfile
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

INPUT = "MFCC/test02/test01"
SAMPLERATE = 9524

mcu_data_info = {
  "Audio": 0,
  "Magnitude": 1,
  "LMS": 2,
  "MFCCs": 3
}
mcu_data = [[] for i in range(len(list(mcu_data_info)))]

# Read in data
raw = pd.read_csv(INPUT + ".log", header=None, encoding='unicode_escape')  

noisy_items = 0
typeidx = 0
for item in list(raw[0]):
  try:
    mcu_data[typeidx].append(np.float(item))
  except ValueError:
    if item in mcu_data_info.keys():
      typeidx += 1
    else:
      noisy_items += 1
        
print("Num noisy items: {}".format(noisy_items))




audio = np.array(mcu_data[0]).astype(np.float32)

### Calculate LMS using tensorflow
sample_rate = 9524.0
lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 4700.0, 64
frame_length = 1024
num_mfcc = 13

stfts = tf.signal.stft(tf.convert_to_tensor(audio[np.newaxis, ...]), 1024, 1024, 1024)[:,:,:-1]
spectrograms = tf.abs(stfts) 
spectrograms = tf.reshape(spectrograms, (spectrograms.shape[0],spectrograms.shape[1],-1))
num_spectrogram_bins = stfts.shape[-1]
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
  num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
  upper_edge_hertz)
mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1) # 512, 64
log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)


# Plotting
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=False)

fig.set_figheight(15)
fig.set_figwidth(20)

# Audio
xaxis = np.arange(len(mcu_data[0]))
ax1.plot(xaxis, mcu_data[0])
ax1.set_xlabel("sample", fontsize=18)
ax1.set_ylabel("Audio", fontsize=10)


# Mag
xaxis = np.arange(len(mcu_data[1]))
ax2.plot(xaxis, np.array(mcu_data[1]), color="blue")
ax2.plot(xaxis, spectrograms.numpy().flatten(), color="red")
ax2.set_xlabel("sample", fontsize=18)
ax2.set_ylabel("Mag", fontsize=10)

# LMS
xaxis = np.arange(len(mcu_data[2]))
ax3.plot(xaxis, mcu_data[2], color="blue")
ax3.plot(xaxis, log_mel_spectrograms.numpy().flatten(), color="red")
ax3.set_xlabel("sample", fontsize=18)
ax3.set_ylabel("LMS", fontsize=10)

# MFCCs
xaxis = np.arange(len(mcu_data[3]))
ax4.plot(xaxis, mcu_data[3], color="blue")
ax4.plot(xaxis, mfccs.numpy().flatten(), color="red")
ax4.set_xlabel("sample", fontsize=18)
ax4.set_ylabel("MFCCs", fontsize=10)
ax4.set_ylim([np.min(np.array(mcu_data[3][:30])), np.max(np.array(mcu_data[3][:30]))])

plt.tight_layout()
plt.show()


