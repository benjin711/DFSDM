# MFCC feature extraction and Network training
# In this notebook you will go through an example flow of processing audio data, complete with feature extraction and training.
# Make sure you read the instructions on the exercise sheet and follow the task order.

# Task 1: Load and resample train and test data
import json
import numpy as np
from scipy.io import wavfile
import scipy.signal as sps
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tqdm import tqdm
import os
import pickle

def resample_wav(old_fs, new_fs, data):
    number_new_samples = int(data.shape[0] * float(new_fs)/float(old_fs))
    data = sps.resample(data, number_new_samples)
    return data

def load_data():
    x_train_list = []
    y_train_list = []

    x_test_list = []
    y_test_list = []

    totalSliceLength = 10 # Length to stuff the signals to, given in seconds

    # trainsize = len(traindata) # Number of loaded training samples
    # testsize = len(testdata) # Number of loaded testing samples

    trainsize = 1000 # Number of loaded training samples
    testsize = 100 # Number of loaded testing samples


    old_fs = 16000 # Sampling rate of the samples
    new_fs = 9524 # Resampling rate
    segmentLength = 1024 # Number of samples to use per segment

    sliceLength = int(totalSliceLength * new_fs / segmentLength)*segmentLength

    print("Num amplitudes per sample {}".format(sliceLength))
    print("Num non overlapping windows {}".format(int(totalSliceLength * new_fs / segmentLength)))

    for i in tqdm(range(trainsize)): 
        fs, train_sound_data = wavfile.read(DataSetPath+traindata[i]['audio_file_path']) # Read wavfile to extract amplitudes

        x_train = train_sound_data.copy() # Get a mutable copy of the wavfile

        _x_train = resample_wav(old_fs, new_fs, x_train)

        _x_train.resize(sliceLength, refcheck=False) # Zero stuff the single to a length of sliceLength
        _x_train = _x_train.reshape(-1,int(segmentLength)) # Split slice into Segments with 0 overlap
        x_train_list.append(_x_train.astype(np.float32)) # Add segmented slice to training sample list, cast to float so librosa doesn't complain
        y_train_list.append(traindata[i]['is_hotword']) # Read label 

    for i in tqdm(range(testsize)):
        fs, test_sound_data = wavfile.read(DataSetPath+testdata[i]['audio_file_path'])
        x_test = test_sound_data.copy()
        _x_test = resample_wav(old_fs, new_fs, x_test)
        _x_test.resize(sliceLength, refcheck=False)
        _x_test = _x_test.reshape((-1,int(segmentLength)))
        x_test_list.append(_x_test.astype(np.float32))
        y_test_list.append(testdata[i]['is_hotword'])

    x_train = tf.convert_to_tensor(np.asarray(x_train_list))
    y_train = tf.convert_to_tensor(np.asarray(y_train_list))

    x_test = tf.convert_to_tensor(np.asarray(x_test_list))
    y_test = tf.convert_to_tensor(np.asarray(y_test_list))

    return x_train, y_train, x_test, y_test

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

def construct_model():
  model = tf.keras.models.Sequential()

  #model.add(layers.InputLayer(input_shape=(train_set.shape[1],train_set.shape[2],train_set.shape[3]), batch_size=(batchSize)))
  model.add(layers.Conv2D(filters=3,kernel_size=(3,3),padding="same",input_shape=(train_set[0].shape)))
  model.add(layers.BatchNormalization())
  model.add(layers.Activation('relu'))

  model.add(layers.Conv2D(filters=16,kernel_size=(3,3),strides=(2,2),padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.Activation('relu'))

  model.add(layers.MaxPool2D((2,2)))

  model.add(layers.Conv2D(filters=32,kernel_size=(3,3),strides=(2,2),padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.Activation('relu'))

  model.add(layers.MaxPool2D((2,2)))

  model.add(layers.Conv2D(filters=48,kernel_size=(3,3),padding='same',strides=(2,2)))
  model.add(layers.BatchNormalization())
  model.add(layers.Activation('relu'))

  model.add(layers.GlobalAveragePooling2D())

  model.add(layers.Flatten())

  model.add(layers.Dense(8, kernel_regularizer=(regularizers.l1(0))))
  model.add(layers.Activation('relu'))

  model.add(layers.Dense(2))
  model.add(layers.Activation('softmax'))

  return model

def hex_to_c_array(hex_data, var_name):
    # Function: Convert some hex value into an array for C programming

    c_str = ''

    # Create header guard
    c_str += '#ifndef ' + var_name.upper() + '_H\n'
    c_str += '#define ' + var_name.upper() + '_H\n\n'

    # Add array length at top of file
    c_str += '\nunsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'

    # Declare C variable
    c_str += 'unsigned char ' + var_name + '[] = {'
    hex_array = []
    for i, val in enumerate(hex_data) :

        # Construct string from hex
        hex_str = format(val, '#04x')

        # Add formatting so each line stays within 80 characters
        if (i + 1) < len(hex_data):
            hex_str += ','
        if (i + 1) % 12 == 0:
            hex_str += '\n '
        hex_array.append(hex_str)

    # Add closing brace
    c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'

    # Close out header guard
    c_str += '#endif //' + var_name.upper() + '_H'

    return c_str  

def print_input_output_details(input_details, output_details):
  print("== Input details ==")
  print("name:", input_details[0]['name'])
  print("shape:", input_details[0]['shape'])
  print("type:", input_details[0]['dtype'])

  print("\n== Output details ==")
  print("name:", output_details[0]['name'])
  print("shape:", output_details[0]['shape'])
  print("type:", output_details[0]['dtype'])

PICKLED_INPUT_OUTPUT = True

SR = 9524.0
LOWER_EDGE_HERTZ, UPPER_EDGE_HERTZ, NUM_MEL_BINS = 80.0, 4700.0, 64
FRAME_LENGTH = 1024
NUM_MFCC = 13

if (not PICKLED_INPUT_OUTPUT):
  DataSetPath = os.path.join("..", "Data", "hey_snips_kws_4.0", "hey_snips_research_6k_en_train_eval_clean_ter/")

  with open(DataSetPath+"train.json") as jsonfile:
      traindata = json.load(jsonfile)

  with open(DataSetPath+"test.json") as jsonfile:
      testdata = json.load(jsonfile)

  x_train, y_train, x_test, y_test = load_data()

  # Task 2: Calculate MFCCs from each input sample 
  x_train_mfcc = compute_mfccs(x_train, SR, LOWER_EDGE_HERTZ, UPPER_EDGE_HERTZ, NUM_MEL_BINS, FRAME_LENGTH, NUM_MFCC)
  x_test_mfcc = compute_mfccs(x_test, SR, LOWER_EDGE_HERTZ, UPPER_EDGE_HERTZ, NUM_MEL_BINS, FRAME_LENGTH, NUM_MFCC)

  # Task 3: Data Normalization & Training & Evaluation

  train_set = (x_train_mfcc/512 + 0.5)
  train_labels = y_train

  test_set = (x_test_mfcc/512 + 0.5)
  test_labels = y_test

  # Pickle the correctly formatted input output
  input_output = (train_set, train_labels, test_set, test_labels)
  PATHTODUMP = os.path.join("..", "Data", "input_output.pkl")
  with open(PATHTODUMP, 'wb') as f:
      pickle.dump(input_output, f) 

###########################
# Unpickle the correctly formatted input output
pickle_off = open(os.path.join("..", "Data", "input_output.pkl"),"rb")
train_set, train_labels, test_set, test_labels = pickle.load(pickle_off)

batchSize = 10
epochs = 1

model = construct_model()
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
history = model.fit(train_set, train_labels, batchSize, epochs, validation_split=0.1, workers=3)
print(model.summary())
score = model.evaluate(test_set, test_labels)
print("Score: {}".format(score))

# Task 4: Store model as keras model and tflite model
MODELS_FOLDER = os.path.join("Network", "Models")
contents = os.listdir(MODELS_FOLDER)
integers = [int(_) for _ in contents if _.isdigit()]
if integers:
  next_model_folder = "{:02d}".format(np.max(np.array(integers, dtype='i4')) + 1)
else:
  next_model_folder = "{:02d}".format(0)

next_model_folder_path = os.path.join(MODELS_FOLDER, "{}".format(next_model_folder))
os.makedirs(next_model_folder_path, exist_ok = True)

# Save parameters of the MFCC calculation
MFCC_PARAMS ={
"sample_rate" : SR,
"lower_edge_hertz" : LOWER_EDGE_HERTZ,
"upper_edge_hertz" : UPPER_EDGE_HERTZ,
"num_mel_bins" : NUM_MEL_BINS,
"frame_length": FRAME_LENGTH,
"num_mfcc": NUM_MFCC
}
with open(os.path.join(next_model_folder_path, "MFCC{}_params.json".format(next_model_folder)), "w") as outfile:
  json.dump(MFCC_PARAMS, outfile)

# Save .h5 model
model.save(os.path.join(next_model_folder_path, "MFCCmodel{}.h5".format(next_model_folder)))

# Save .tflite + header
train_set = train_set.numpy()
test_set = test_set.numpy()
train_labels = train_labels.numpy()
test_labels = test_labels.numpy()
tflite_model_name = 'MFCC{}'.format(next_model_folder)
windows_per_sample = int(10 * 9524.0 / 1024)
# Convert Keras model to a tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Convert the model to the TensorFlow Lite format with quantization
quantize = True
if (quantize):
    def representative_dataset():
        for i in range(500):
            yield([train_set[i].reshape(1,windows_per_sample,13,1)])
    # Set the optimization flag.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Enforce full-int8 quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    # Provide a representative dataset to ensure we quantize correctly.
converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

open(os.path.join(next_model_folder_path, tflite_model_name + '.tflite'), 'wb').write(tflite_model)

c_model_name = 'MFCC'
# Write TFLite model to a C source (or header) file
with open(os.path.join(next_model_folder_path, c_model_name + '.h'), 'w') as file:
    file.write(hex_to_c_array(tflite_model, c_model_name))

tflite_interpreter = tf.lite.Interpreter(model_path=os.path.join(next_model_folder_path, tflite_model_name + '.tflite'))
tflite_interpreter.allocate_tensors()
input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()
print_input_output_details(input_details, output_details)

# Task 6: Evaluate tflite model
predictions = np.zeros((len(test_set),), dtype=int)
input_scale, input_zero_point = input_details[0]["quantization"]
for i in range(len(test_set)):
    val_batch = test_set[i]
    val_batch = val_batch / input_scale + input_zero_point
    val_batch = np.expand_dims(val_batch, axis=0).astype(input_details[0]["dtype"])
    tflite_interpreter.set_tensor(input_details[0]['index'], val_batch)
    tflite_interpreter.allocate_tensors()
    tflite_interpreter.invoke()

    tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])
    #print("Prediction results shape:", tflite_model_predictions.shape)
    output = tflite_interpreter.get_tensor(output_details[0]['index'])
    predictions[i] = output.argmax()

sum = 0
for i in range(len(predictions)):
    if (predictions[i] == test_labels[i]):
        sum = sum + 1
accuracy_score = sum / 100
print("Accuracy of quantized to int8 model is {}%".format(accuracy_score*100))
print("Compared to float32 accuracy of {}%".format(score[1]*100))
print("We have a change of {}%".format((accuracy_score-score[1])*100))