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
from keras.layers import LSTM, concatenate, Input, Concatenate, Multiply, Dropout, Subtract, Add, Conv1D
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
from datetime import datetime
import copy

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
        #print('shuffling')
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        return int(np.floor(len(self.the_data) / self.batch_size))

    def check_differences(self, comm1, comm2):
        """ we give here 2 commit names"""
        #print('checking the difference')
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
        #        print('diff left right')
        if diff21.size != 0:
            for k in range(diff21.shape[0]):
                with open(self.path_to_commits + '/' + comm2 + '/' + diff21[k], 'rb') as f:
                    content = pickle.load(f)
        #        print('diff right left')
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
        #print('difference_left_has_length ', len(diff_left), 'difference_right_has_length ', len(diff_right))
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
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
        #print(dt_string+' doing generator iteration \n')
        # print(batch_samples)
        for batch_sample in batch_samples:
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


# https://github.com/prabhnoor0212/Siamese-Network-Text-Similarity/blob/master/quora_siamese.ipynb

def getmodel_1(MAXLEN, units_in_common_layer=64, units_in_dense_layer=100,units_in_common_layer_1=50, units_in_dense_layer_1 = 50, f_activation="tanh"):
    input_1 = Input(shape=(MAXLEN,))
    input_2 = Input(shape=(MAXLEN,))

    common_embed = wvmodel.wv.get_keras_embedding(train_embeddings=True)
    lstm_1 = common_embed(input_1)
    lstm_2 = common_embed(input_2)

    common_lstm = LSTM(units_in_common_layer, return_sequences=True, activation=f_activation)
    vector_1 = common_lstm(lstm_1)


    vector_2 = common_lstm(lstm_2)

    common_lstm2 = LSTM(units_in_common_layer_1, return_sequences=True, activation=f_activation)

    vector_1 = common_lstm2(vector_1)
    vector_2 = common_lstm2(vector_2)

    vector_1 = Flatten()(vector_1)
    vector_2 = Flatten()(vector_2)
    x3 = Subtract()([vector_1, vector_2])
    x3 = Multiply()([x3, x3])

    x1_ = Multiply()([vector_1, vector_1])
    x2_ = Multiply()([vector_2, vector_2])
    x4 = Subtract()([x1_, x2_])

    # https://stackoverflow.com/a/51003359/10650182
    x5 = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([vector_1, vector_2])

    conc = Concatenate(axis=-1)([x5, x4, x3])

    x = Dense(units_in_dense_layer, activation="relu", name='conc_layer')(conc)
    x = Dropout(0.01)(x)
    out = Dense(1, activation="sigmoid", name='out')(x)

    model = Model([input_1, input_2], out)

    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))

    return model


def getmodel_2(MAXLEN, units_in_common_layer=64, units_in_dense_layer=100,units_in_common_layer_1=50, units_in_dense_layer_1 = 50, f_activation="tanh"):
    input_1 = Input(shape=(MAXLEN,))
    input_2 = Input(shape=(MAXLEN,))

    common_embed = wvmodel.wv.get_keras_embedding(train_embeddings=True)
    lstm_1 = common_embed(input_1)
    lstm_2 = common_embed(input_2)

    common_lstm = LSTM(units_in_common_layer, return_sequences=True, activation=f_activation)
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

    x = Dense(units_in_dense_layer, activation="tanh", name='conc_layer')(conc)
    x = Dense(units_in_dense_layer_1, activation="relu", name='second_dense')(x)
    x = Dropout(0.01)(x)
    out = Dense(1, activation="sigmoid", name='out')(x)

    model = Model([input_1, input_2], out)

    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))

    return model

def getmodel_3(MAXLEN, units_in_common_layer=64, units_in_dense_layer=100,units_in_common_layer_1=50, units_in_dense_layer_1 = 50, f_activation="tanh"):
    input_1 = Input(shape=(MAXLEN,))
    input_2 = Input(shape=(MAXLEN,))

    common_embed = wvmodel.wv.get_keras_embedding(train_embeddings=True)
    lstm_1 = common_embed(input_1)
    lstm_2 = common_embed(input_2)

    common_conv1d = Conv1D(units_in_common_layer, kernel_size=5, activation=f_activation)
    vector_1 = common_conv1d(lstm_1)

    vector_2 = common_conv1d(lstm_2)

    common_conv1d2 = Conv1D(units_in_common_layer_1, kernel_size=5, activation=f_activation)

    vector_1 = common_conv1d2(vector_1)
    vector_2 = common_conv1d2(vector_2)

    vector_1 = Flatten()(vector_1)
    vector_2 = Flatten()(vector_2)
    x3 = Subtract()([vector_1, vector_2])
    x3 = Multiply()([x3, x3])

    x1_ = Multiply()([vector_1, vector_1])
    x2_ = Multiply()([vector_2, vector_2])
    x4 = Subtract()([x1_, x2_])

    # https://stackoverflow.com/a/51003359/10650182
    x5 = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([vector_1, vector_2])

    conc = Concatenate(axis=-1)([x5, x4, x3])

    x = Dense(units_in_dense_layer, activation="relu", name='conc_layer')(conc)
    x = Dropout(0.01)(x)
    out = Dense(1, activation="sigmoid", name='out')(x)

    model = Model([input_1, input_2], out)

    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))

    return model


def model_pipeline(the_data, model, name, maxlen, batch_size=4,
                   processed_dirs='git_processed', epochs = 5, model_dirs='trained_models'):
    generator = DataGeneratorSiamese_if_preprocessed(the_data, maxlen=maxlen,
                                               path_to_commits=processed_dirs, wvmodel=wvmodel, batch_size=batch_size,
                                               padding='post',
                                               to_fit=True)
    saving_to = model_dirs+'/'+name
    os.mkdir(saving_to)
    filepath = name+"w-imp-{epoch:02d}-{acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(saving_to+'/'+filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit_generator(generator, epochs=epochs, verbose=2, workers=1, use_multiprocessing=False, callbacks=callbacks_list)
    model.save(saving_to+'/'+name+'_final.hdf5')
    return history


def do_the_grid(the_data, number_of_models,  maxlens, batch_sizes, array_of_common_units, array_of_dense_units, array_of_activation_functions,
                array_of_epochs, array_of_dense_units_1, array_of_common_units_1,  processed_dirs='git_processed', model_dirs='trained_models'):
    mls = np.random.choice(maxlens, number_of_models)
    bts = np.random.choice(batch_sizes, number_of_models)
    cus = np.random.choice(array_of_common_units, number_of_models)
    dus = np.random.choice(array_of_dense_units, number_of_models)
    cus1 = np.random.choice(array_of_common_units_1, number_of_models)
    dus1 = np.random.choice(array_of_dense_units_1, number_of_models)
    afs = np.random.choice(array_of_activation_functions, number_of_models)
    eps = np.random.choice(array_of_epochs, number_of_models)
    global global_name_best
    global global_best_acc
    for model_numb in range(number_of_models):
        name_common = str(mls[model_numb])+'_'+str(bts[model_numb])+'_'+str(cus[model_numb])+'_'+str(cus1[model_numb])+'_'+str(dus[model_numb])+'_'+str(dus1[model_numb])+'_'+afs[model_numb]+'_'+str(eps[model_numb])
        model1 = getmodel_1(MAXLEN=mls[model_numb], units_in_common_layer=cus[model_numb],
                            units_in_dense_layer=dus[model_numb], f_activation=afs[model_numb], units_in_common_layer_1=cus1[model_numb],
                            units_in_dense_layer_1=dus1[model_numb])
        name_1 = 'model_lstmx2_'+name_common
        model2 = getmodel_2(MAXLEN=mls[model_numb], units_in_common_layer=cus[model_numb],
                            units_in_dense_layer=dus[model_numb], f_activation=afs[model_numb], units_in_common_layer_1=cus1[model_numb],
                            units_in_dense_layer_1=dus1[model_numb])
        name_2 = 'model_densex2_' + name_common
        model3 = getmodel_3(MAXLEN=mls[model_numb], units_in_common_layer=cus[model_numb],
                            units_in_dense_layer=dus[model_numb], f_activation=afs[model_numb], units_in_common_layer_1=cus1[model_numb],
                            units_in_dense_layer_1=dus1[model_numb])
        name_3 = 'model_convx2_' + name_common
        history1 = model_pipeline(the_data, model1, name_1, maxlen= mls[model_numb], batch_size=bts[model_numb], processed_dirs=processed_dirs, epochs=eps[model_numb],
                       model_dirs= model_dirs)
        history2 = model_pipeline(the_data, model2, name_2, maxlen= mls[model_numb], batch_size=bts[model_numb], processed_dirs=processed_dirs, epochs=eps[model_numb],
                       model_dirs= model_dirs)
        history3 = model_pipeline(the_data, model3, name_3, maxlen= mls[model_numb], batch_size=bts[model_numb], processed_dirs=processed_dirs, epochs=eps[model_numb],
                       model_dirs= model_dirs)
        try:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
            print(dt_string + 'accuracies ', history1.history['accuracy'][-1], history2.history['accuracy'][-1], history3.history['accuracy'][-1])
            if history1.history['accuracy'][-1] > global_best_acc:
                global_best_acc = copy.deepcopy(history1.history['accuracy'][-1])
                global_name_best = copy.deepcopy(name_1)
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
                print(dt_string+name_1+' have improved_best_acc to ', global_best_acc)
            if history2.history['accuracy'][-1] > global_best_acc:
                global_best_acc = copy.deepcopy(history2.history['accuracy'][-1])
                global_name_best = copy.deepcopy(name_2)
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
                print(dt_string+name_2+' have improved_best_acc to ', global_best_acc)
            if history3.history['accuracy'][-1] > global_best_acc:
                global_best_acc = copy.deepcopy(history3.history['accuracy'][-1])
                global_name_best = copy.deepcopy(name_3)
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
                print(dt_string+name_3+' have improved_best_acc to ', global_best_acc)
        except:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
            print('\n'+dt_string+' ERROR WITH GETTING ACCURACY ON MODELS ' + name_common)
            continue






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
for numb in range(the_dataq.shape[0]):
    if the_dataq[numb, 2] > 1:
        the_dataq[numb, 2] = 1

number_of_models = 120
maxlens = np.array([400, 600, 700])
batch_sizes = np.array([16, 32])
array_of_common_units =np.array([40, 50, 80])
array_of_dense_units = np.array([80, 100, 128])
array_of_activation_functions = np.array(['tanh'])
array_of_epochs = np.array([10, 15])
array_of_dense_units_1 = np.array([40, 50])
array_of_common_units_1 = np.array([20, 30, 40])
processed_dirs = 'git_preprocessed'
model_dirs = 'trained_models_1'
global_name_best = ''
global_best_acc = 0
do_the_grid(the_dataq, number_of_models=number_of_models,  maxlens=maxlens, batch_sizes=batch_sizes,
            array_of_common_units=array_of_common_units, array_of_dense_units=array_of_dense_units,
            array_of_common_units_1=array_of_common_units_1, array_of_dense_units_1=array_of_dense_units_1,
            array_of_activation_functions=array_of_activation_functions, array_of_epochs=array_of_epochs,
            processed_dirs=processed_dirs, model_dirs=model_dirs)

print(global_name_best,' has a accuracy ', global_best_acc)

#gen = DataGeneratorSiamese_if_preprocessed(the_dataq, maxlen=MAXLEN,
#                                            path_to_commits=processed_dirs, wvmodel=wvmodel, batch_size=4, padding='post',
#                                           to_fit=True)

#model.fit_generator(gen, epochs=2, verbose=2, workers=1, use_multiprocessing=False)