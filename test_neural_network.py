import numpy as np
import os
import sys
import shutil
import pandas as pd
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, concatenate, Input, Concatenate, Multiply, Dropout, Subtract, Add, Conv2D
from keras.models import Model
from keras.layers.core import Lambda
from keras.layers import  GRU, Activation, PReLU, Bidirectional
from keras.optimizers import Adam, Adadelta
from keras.callbacks import ModelCheckpoint
import gensim.models
from keras.models import Sequential
from keras.losses import sparse_categorical_crossentropy, mean_squared_error, cosine_proximity
from keras import backend as K
from keras.utils import Sequence
import pickle


def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)


class DataGeneratorSiamese_if_preprocessed(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, the_data, maxlen, path_to_commits, wvmodel, batch_size=4, padding='post', to_fit=True, shuffle=True):
        """Initialization
        """
        self.the_data = np.array(the_data)
        self.maxlen = maxlen
        # PATH TO COMMITS W/O / IN THE END
        self.path_to_commits = path_to_commits
        self.shuffle = shuffle
        self.wvmodel = wvmodel
        self.batch_size = batch_size
        self.to_fit = to_fit
        self.list_of_all_commits = os.listdir(path_to_commits)
        self.padding = padding

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        self.indexes = np.arange(len(self.the_data))
        print('shuffling')
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        return int(np.floor(len(self.the_data) / self.batch_size))

    def check_differences(self, comm1, comm2):
        """ we give here 2 commit names"""
        print('checking the difference')
        diff_left = np.array([])
        diff_right = np.array([])

        comm1_files = os.listdir(self.path_to_commits + '/' + comm1)
        comm2_files = os.listdir(self.path_to_commits + '/' + comm2)
        diff12 = np.setdiff1d(comm1_files, comm2_files)
        diff21 = np.setdiff1d(comm2_files, comm1_files)
        if diff12.size != 0:
            for k in range(diff12.shape[0]):
                with open(self.path_to_commits + '/' + comm1 + '/' + diff12[k], 'rb') as f:
                    content = pickle.load(f)
                diff_left = np.concatenate([diff_left, content])
                print('diff left right')
        if diff21.size != 0:
            for k in range(diff21.shape[0]):
                with open(self.path_to_commits + '/' + comm2 + '/' + diff21[k], 'rb') as f:
                    content = pickle.load(f)
                print('diff right left')
                diff_right = np.concatenate([diff_right, content])

        same_files = np.setdiff1d(comm1_files, diff12)
        for filename_n in range(same_files.shape[0]):
            with open(self.path_to_commits + '/' + comm1 + '/' + same_files[filename_n], 'rb') as f:
                content_left = pickle.load(f)
            with open(self.path_to_commits + '/' + comm2 + '/' + same_files[filename_n], 'rb') as f:
                content_right = pickle.load(f)
            if not np.array_equal(content_left, content_right):
                diff_left = np.concatenate([diff_left, content_left])
                diff_right = np.concatenate([diff_right, content_right])
        return diff_left, diff_right

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """

        batch_samples = self.the_data[self.batch_size * index: self.batch_size * (index + 1)]
        commits_a = []
        commits_b = []
        y_train = []
        print('doing generator iteration \n')
        # print(batch_samples)
        for batch_sample in batch_samples:
            # RANDOM LABEL HELLO HERE
            commit_a = batch_sample[0]
            commit_b = batch_sample[1]
            label = batch_sample[2]
            diff_left, diff_right = self.check_differences(commit_a, commit_b)
            commits_a.append(diff_left)
            commits_b.append(diff_right)
            y_train.append(label)
        data_a = pad_sequences(np.array(commits_a), maxlen=self.maxlen)
        data_b = pad_sequences(np.array(commits_b), maxlen=self.maxlen)
        if self.to_fit:
            return [data_a, data_b], y_train
        else:
            return [data_a, data_b]

    #def on_epoch_end(self):
    #    """Updates indexes after each epoch
    ##    """
    #    self.indexes = np.arange(len(self.the_data))
    #    print('shuffling')
    #    if self.shuffle == True:
    #        np.random.shuffle(self.indexes)


df = pd.read_csv('no_zeros.csv')
needed = df.columns[:2]
needed = needed.tolist()
needed.append(df.columns[-1])
needed = np.array(needed)
dd = df[needed]
data = dd.groupby(['Commit A', 'Commit B'], as_index=False).sum()
the_data = data.values
commits = data[data.columns[:2]].values
labels = data[data.columns[-1]].values
wvmodel = gensim.models.Word2Vec.load('word2vec.model')
path_to_dirs = 'git_txts'
dirs = os.listdir(path_to_dirs)
the_dataq=the_data.copy()
processed_dirs = 'git_preprocessed'
MAXLEN=30

# https://github.com/prabhnoor0212/Siamese-Network-Text-Similarity/blob/master/quora_siamese.ipynb

input_1 = Input(shape=(MAXLEN,))
input_2 = Input(shape=(MAXLEN,))

common_embed = wvmodel.wv.get_keras_embedding(train_embeddings=False)
lstm_1 = common_embed(input_1)
lstm_2 = common_embed(input_2)

common_lstm = LSTM(64, return_sequences=True, activation="relu")
vector_1 = common_lstm(lstm_1)
vector_1 = Flatten()(vector_1)

vector_2 = common_lstm(lstm_2)
vector_2 = Flatten()(vector_2)

x3 = Subtract()([vector_1, vector_2])
x3 = Multiply()([x3, x3])

x1_ = Multiply()([vector_1, vector_1])
x2_ = Multiply()([vector_2, vector_2])
x4 = Subtract()([x1_, x2_])

# https://stackoverflow.com/a/51003359/10650182
x5 = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([vector_1, vector_2])

conc = Concatenate(axis=-1)([x5, x4, x3])

x = Dense(100, activation="relu", name='conc_layer')(conc)
x = Dropout(0.01)(x)
out = Dense(1, activation="sigmoid", name='out')(x)

model = Model([input_1, input_2], out)

model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))


gen = DataGeneratorSiamese_if_preprocessed(the_dataq, maxlen=MAXLEN,
                                            path_to_commits=processed_dirs, wvmodel=wvmodel, batch_size=4, padding='post',
                                           to_fit=True)

model.fit_generator(gen, epochs=2, verbose=2, workers=1, use_multiprocessing=False)