import json
import numpy as np
from scipy.io import wavfile
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tqdm import tqdm
import os
import pickle
import sys

def mat_to_c_array_1(numpy_data, file_name):

    c_str = ''

    # Create header guard
    c_str += '#ifndef ' + file_name.upper() + '_H\n'
    c_str += '#define ' + file_name.upper() + '_H\n\n'


    # Add array size
    c_str += '\nint ' + file_name + '_size = ' + str(numpy_data.size) + ';\n'
    # Add number of columns
    c_str += '\nint ' + file_name + '_cols = ' + str(numpy_data.shape[0]) + ';\n'
    # Add number of rows
    c_str += '\nint ' + file_name + '_rows = ' + str(numpy_data.shape[1]) + ';\n\n'

    # Declare C variable
    c_str += 'float ' + 'file_name' + '_raw' + '[] = {'
    array = []
    list_data = numpy_data.flatten().tolist()
    for i, val in enumerate(list_data) :

        # Construct string 
        dec_str = "{:.6f}".format(val)

        if (i + 1) < len(list_data):
            dec_str += ','
        if (i + 1) % 12 == 0:
            dec_str += '\n '
        array.append(dec_str)

    # Add closing brace
    c_str += '\n ' + format(' '.join(array)) + '\n};\n\n'

    # Close out header guard
    c_str += '#endif //' + file_name.upper() + '_H'

    return c_str

def mat_to_c_array_2(numpy_data, file_name, num_mel_bins):
    c_str = ''

    # Create header guard
    c_str += '#ifndef ' + file_name.upper() + '_H\n'
    c_str += '#define ' + file_name.upper() + '_H\n\n'

    # Includes
    c_str += '#include <arm_math.h>\n\n' 

    # Declare C variable
    c_str += 'float32_t ' + 'file_name' + '[] = {'

    # Number of nonzero elements
    num_nonzero = np.sum(numpy_data > 0)

    array = []
    comma_counter = 0
    for row in numpy_data.T:
      non_zero = np.nonzero(row)[0]
      row_str = "{:.0f}".format(non_zero.shape[0])
      row_str += ', '

      comma_counter += 1
  
      for idx in non_zero.tolist():
        row_str += "{:.0f}, {:.6f}".format(idx, row[idx]) # index in row, value of row 

        comma_counter += 2

        if (comma_counter + 1) < num_nonzero * 2 + num_mel_bins:
            row_str += ', '

      row_str += '\n '

      array.append(row_str)
    
    # Add closing brace
    c_str += '\n ' + format(' '.join(array)) + '\n};\n\n'

    # Add array size
    c_str += '\nint ' + file_name + '_size = ' + str(np.sum(numpy_data > 0) * 2 + num_mel_bins) + ';\n\n'
    
    # Close out header guard
    c_str += '#endif //' + file_name.upper() + '_H'

    return c_str


def compute_mfccs(tensor):
    sample_rate = 9524.0
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 4700.0, 64
    frame_length = 1024
    num_mfcc = 13

    stfts = tf.signal.stft(tensor, frame_length=frame_length, frame_step=frame_length, fft_length=frame_length)[:,:,:,:-1] # cause somehow the arm only calcs 512 elements
    spectrograms = tf.abs(stfts)
    spectrograms = tf.reshape(spectrograms, (spectrograms.shape[0],spectrograms.shape[1],-1))
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
      upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1) # 512, 80
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :num_mfcc]
    return tf.reshape(mfccs, (mfccs.shape[0],mfccs.shape[1],mfccs.shape[2],-1)), linear_to_mel_weight_matrix.numpy()

pickle_off = open('x_train.txt', "rb")
x_train = pickle.load(pickle_off)
x_train_mfcc, linear_to_mel_weight_matrix = compute_mfccs(x_train)

# matrix_name = 'linear_to_mel_weight_matrix'
# with open(matrix_name + '.h', 'w') as file:
#     file.write(mat_to_c_array_1(linear_to_mel_weight_matrix, matrix_name))



list_name = 'linear_to_mel_weight_list'
with open(list_name + '.h', 'w') as file:
  file.write(mat_to_c_array_2(linear_to_mel_weight_matrix, list_name, 64))




print("Done")