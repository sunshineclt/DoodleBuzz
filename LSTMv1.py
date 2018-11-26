import datetime as dt
import os
from platform import platform

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from ast import literal_eval
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.sequence import pad_sequences
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv1D, LSTM, Dense, Dropout, Bidirectional

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


def f2cat(filename: str) -> str:
    return filename.split('.')[0]


def list_all_categories():
    files = os.listdir(os.path.join(INPUT_DIR, 'train_simplified'))
    return sorted([f2cat(f) for f in files], key=str.lower)


def apk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)


def mapk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


start = dt.datetime.now()

debug = False
if debug:
    STEPS = 200
    val_steps = 10
else:
    STEPS = 800
    val_steps = 100

STROKE_COUNT = 100
EPOCHS = 20
batchsize = 256

if 'Darwin' in platform():
    DP_DIR = './shuffle-csvs/'
    INPUT_DIR = './input/'
else:
    DP_DIR = '/big/shuffle-csvs/'
    INPUT_DIR = '/big/'

NCSVS = 100
NCATS = 340
np.random.seed(seed=1987)
tf.set_random_seed(seed=1987)


def _stack_it(raw_strokes):
    """preprocess the string and make
    a standard Nx3 stroke vector"""
    stroke_vec = literal_eval(raw_strokes)  # string->list
    # unwrap the list
    in_strokes = [(xi, yi, i)
                  for i, (x, y) in enumerate(stroke_vec)
                  for xi, yi in zip(x, y)]
    c_strokes = np.stack(in_strokes)
    # replace stroke id with 1 for continue, 2 for new
    c_strokes[:, 2] = [1] + np.diff(c_strokes[:, 2]).tolist()
    c_strokes[:, 2] += 1  # since 0 is no stroke
    # pad the strokes with zeros
    return pad_sequences(c_strokes.swapaxes(0, 1),
                         maxlen=STROKE_COUNT,
                         padding='post').swapaxes(0, 1)


SHUFFLE_REPEAT = 16


def _shuffle_stack_it(raw_strokes):
    stroke_vec = literal_eval(raw_strokes)  # string->list
    result = []
    for i in range(SHUFFLE_REPEAT):
        stroke_num = len(stroke_vec)
        shuffle_list = np.linspace(0, stroke_num-1, stroke_num)
        np.random.shuffle(shuffle_list)
        in_strokes = [(xi, yi, shuffle_list[i], i1)
                  for i, (x, y) in enumerate(stroke_vec)
                  for i1, (xi, yi) in enumerate(zip(x, y))]
        dtype = [('x', np.int), ('y', np.int), ('index1', np.int), ('index2', np.int)]
        c_strokes = np.array(in_strokes, dtype=dtype)
        c_strokes.sort(axis=0, order=['index1','index2'])
        c_strokes = c_strokes[["x", 'y', 'index1']].copy()
        c_strokes = c_strokes.view((int, len(c_strokes.dtype.names)))
        # replace stroke id with 1 for continue, 2 for new
        c_strokes[:, 2] = [1] + np.diff(c_strokes[:, 2]).tolist()
        c_strokes[:, 2] += 1  # since 0 is no stroke
        # pad the strokes with zeros
        result.append(pad_sequences(c_strokes.swapaxes(0, 1),
                      maxlen=STROKE_COUNT,
                      padding='post').swapaxes(0, 1))
    return result


def image_generator_xd(batchsize, ks, data_augmentation=False):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize / SHUFFLE_REPEAT):
                if data_augmentation:
                    x1 = df['drawing'].map(_shuffle_stack_it)
                    x2 = np.concatenate(x1, axis=0)
                    y = df.y
                    y = np.repeat(y, SHUFFLE_REPEAT)
                    y = keras.utils.to_categorical(y, num_classes=NCATS)
                    yield x2, y
                else:
                    df['drawing'] = df['drawing'].map(_stack_it)
                    x2 = np.stack(df['drawing'], 0)
                    y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                    yield x2, y


def df_to_image_array_xd(df):
    df['drawing'] = df['drawing'].map(_stack_it)
    x2 = np.stack(df['drawing'], 0)
    return x2


def df_to_image_array_shuffle_xd(df):
    df['drawing'] = df['drawing'].map(_shuffle_stack_it)
    x2 = np.stack(df['drawing'], 0)
    return x2


valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=34000)
x_valid = df_to_image_array_xd(valid_df)
y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
print(x_valid.shape, y_valid.shape)
print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024. ** 3))

train_datagen = image_generator_xd(batchsize=batchsize, ks=range(NCSVS - 1), data_augmentation=True)

# if len(get_available_gpus())>0:
# https://twitter.com/fchollet/status/918170264608817152?lang=en
#    from keras.layers import CuDNNLSTM as LSTM # this one is about 3x faster on GPU instances
stroke_read_model = Sequential()
stroke_read_model.add(BatchNormalization(input_shape=(None,) + (3,)))
# filter count and length are taken from the script https://github.com/tensorflow/models/blob/master/tutorials/rnn/quickdraw/train_model.py
stroke_read_model.add(Conv1D(256, (5,), activation='relu'))
stroke_read_model.add(Dropout(0.2))
stroke_read_model.add(Conv1D(256, (5,), activation='relu'))
stroke_read_model.add(Dropout(0.2))
stroke_read_model.add(Conv1D(256, (3,), activation='relu'))
stroke_read_model.add(Dropout(0.2))
stroke_read_model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)))
stroke_read_model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)))
stroke_read_model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=False)))
stroke_read_model.add(Dense(512, activation='relu'))
stroke_read_model.add(Dropout(0.2))
stroke_read_model.add(Dense(NCATS, activation='softmax'))
stroke_read_model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['categorical_accuracy', top_3_accuracy])
stroke_read_model.summary()

weight_path = "{}_weights.best.hdf5".format('stroke_lstm_bidirectional_relu')

if os.path.exists(weight_path):
    print("Loading Model!")
    stroke_read_model.load_weights(weight_path)
    print("Model Loaded!")

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,
                                   verbose=1, mode='auto', min_delta=0.0001, cooldown=3, min_lr=1e-5)
early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=3)
callbacks_list = [checkpoint, early, reduceLROnPlat]
hist = stroke_read_model.fit_generator(train_datagen, steps_per_epoch=STEPS, epochs=EPOCHS, verbose=1,
                                       validation_data=(x_valid, y_valid),
                                       callbacks=callbacks_list)

if 'Darwin' in platform():
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv('hist_training.csv')
    hist_df.index = np.arange(1, len(hist_df) + 1)
    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(16, 10))
    axs[0].plot(hist_df.val_top_3_accuracy, lw=5, label='Validation Accuracy')
    axs[0].plot(hist_df.top_3_accuracy, lw=5, label='Training Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].grid()
    axs[0].legend(loc=0)
    axs[1].plot(hist_df.val_loss, lw=5, label='Validation MLogLoss')
    axs[1].plot(hist_df.loss, lw=5, label='Training MLogLoss')
    axs[1].set_ylabel('MLogLoss')
    axs[1].set_xlabel('Epoch')
    axs[1].grid()
    axs[1].legend(loc=0)
    fig.savefig('hist.png', dpi=300)
    plt.show()

valid_predictions = stroke_read_model.predict(x_valid, batch_size=batchsize, verbose=1)
map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)
print('Map3: {:.3f}'.format(map3))
# lstm_results = stroke_read_model.evaluate(x_valid, y_valid, batch_size=4096)
# print('Accuracy: %2.1f%%, Top 3 Accuracy %2.1f%%' % (100 * lstm_results[1], 100 * lstm_results[2]))

test = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))
x_test = df_to_image_array_xd(test)
print(test.shape, x_test.shape)
print('Test array memory {:.2f} GB'.format(x_test.nbytes / 1024. ** 3))

test_predictions = stroke_read_model.predict(x_test, verbose=True, batch_size=batchsize)

top3 = preds2catids(test_predictions)
cats = list_all_categories()
id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
top3cats = top3.replace(id2cat)

test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
submission = test[['key_id', 'word']]
now = dt.datetime.now()
submission.to_csv('lstm_submission_{}-{}_{}:{}_{}.csv'.format(now.month, now.day, now.hour, now.minute, int(map3 * 10**4)), index=False)
submission.head()

end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))
