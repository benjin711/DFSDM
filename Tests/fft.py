import numpy as np
from scipy.io import wavfile
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

INPUT = "FFT/test05/test01"
OUTPUT = "FFT/test05/test02"
SAMPLERATE = 9524


audio = pd.read_csv(INPUT + ".log", header=None, encoding='unicode_escape')
cfft = pd.read_csv(OUTPUT + ".log", header=None, encoding='unicode_escape')
audio_fil = []
cfft_fil = []    

noisy_items = 0
for item in list(audio[0]):
    try:
        audio_fil.append(np.float(item))
    except ValueError:
        noisy_items += 1

for item in list(cfft[0]):
    try:
        cfft_fil.append(np.float(item))
    except ValueError:
        noisy_items += 1
        
print("Num noisy items: {}".format(noisy_items))
    
audio_fil = np.array(audio_fil).astype(np.float)
cfft_real_fil = np.array(cfft_fil[0::2]).astype(np.float)
cfft_cmplx_fil = np.array(cfft_fil[1::2]).astype(np.float)

### Calculate FFT using tensorflow
stfts = tf.signal.stft(tf.convert_to_tensor(audio_fil[np.newaxis, ...]), 1024, 1024, 1024).numpy()[0,0]
rstfts = stfts.real
istfts = stfts.imag

xaxis = np.arange(audio_fil.shape[0])

# Plotting
fig, (ax2, ax3, ax4) = plt.subplots(nrows=3, ncols=1, sharex=True)
# Audio
#ax1.plot(xaxis, audio_fil)

# Cosine
ax2.plot(xaxis[:512], cfft_real_fil, color="blue")
ax2.plot(xaxis[:513], rstfts[:513], color="red")

# Phase
ax3.plot(xaxis[:512], cfft_cmplx_fil, color="blue")
ax3.plot(xaxis[:513], istfts[:513], color="red")

# # Magnitude
mag_cfft = np.sqrt(np.square(cfft_real_fil) + np.square(cfft_cmplx_fil))
mag_stfts = np.sqrt(np.square(rstfts) + np.square(istfts))

ax4.plot(xaxis[:512], mag_cfft, color="blue")
ax4.plot(xaxis[:512], mag_stfts[:512], color="red")


# ax1.set_xlabel("x[m]", fontsize=18)
#ax1.set_ylabel("audio", fontsize=10)
ax2.set_ylabel("arm rfft", fontsize=10)
ax3.set_ylabel("arm ifft", fontsize=10)
ax4.set_ylabel("mag", fontsize=10)

plt.tight_layout()
plt.show()