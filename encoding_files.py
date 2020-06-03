import pickle
import gensim.models
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
import os
import sys
import shutil
import pandas as pd
from datetime import datetime


def preprocess_commits_and_prep_for_sequencing(the_data, list_of_all_commits, path_to_commits, path_to_save):
    needed_commits = np.unique(np.concatenate([the_data[:, 1], the_data[: ,0]]))

    #print(needed_commits)
    for n_commit in range (needed_commits.shape[0]):
        print('prerpocessing ', n_commit)
        commita = needed_commits[n_commit]
        if os.path.exists(path_to_save +commita):
            shutil.rmtree(path_to_save +commita)
        os.mkdir(path_to_save +commita)
        in_it = False
        for comm in list_of_all_commits:
            if comm.startswith(commita):
                commit = comm
                print('found')
                in_it = True
        if not in_it:
            print('ERROR NOT FOUND ' + comm)
            continue
        files = os.listdir(path_to_commits + '/' +commit + '/')
        temp = []
        for file in files:
            with open(path_to_commits +'/' +commit +'/' +file) as f:
                content = f.read()
                # print(content)
            arr = np.array(text_to_word_sequence(content))
            print('padding sequences')

            array_of_word_lists = arr
            source_word_indices = []
            # print(arr)
            for i in range(len(array_of_word_lists)):
                word = array_of_word_lists[i]
                # print(word)
                # print('doing word number', j,' out of ', len(array_of_word_lists[i]))
                if word in wvmodel.wv.vocab:
                    word_index = wvmodel.wv.vocab[word].index
                    source_word_indices.append(word_index)
                    # print('AAA')
                # else:
            # Do something. For example, leave it blank or replace with padding character's index.
            # source_word_indices[i].append(padding_index)

            source = np.array(source_word_indices)
            # print(source)
            with open(path_to_save + commita + '/' + file + '.pickle', 'wb') as f:
                pickle.dump(source, f)
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
                print('\n'+ dt_string + ' ' + commita + '/' + file, ' complete')

        # temp.append(content)
    return


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
processed_dirs = 'git_preprocessed/'
for numb in range(the_dataq.shape[0]):
    if the_dataq[numb, 2] > 1:
        the_dataq[numb, 2] = 1

preprocess_commits_and_prep_for_sequencing(the_dataq, os.listdir(path_to_dirs),path_to_dirs,
                                                           processed_dirs)


