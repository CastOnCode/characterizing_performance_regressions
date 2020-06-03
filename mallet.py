import os
from datetime import datetime

def get_directories(basepath):
    res = []
    for entry in os.listdir(basepath):
        if os.path.isdir(os.path.join(basepath, entry)) and entry[0]!='.':
            res.append(entry)
    return res


def mallet_all_subdirs(dirs, path_to_txts, path_to_mallet_outs, path_to_end_mallet_topics, path_to_end_mallet_keys, mallet_call, num_topics):
    with open('log_malleting.txt', 'w+') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
        f.write('\n' + dt_string + ' starting_to_log')
    for k in range(len(dirs)):

        dr = dirs[k]
        with open('log_malleting.txt', 'a') as f:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
            f.write('\n' + dt_string + ' malleting_directory ' + dr)
        d = path_to_txts+dr+'/'
        #print(mallet_call+' import-dir --input ' + d + ' --output ' + path_to_mallet_outs + dr +'.mallet --keep-sequence')
        os.system(mallet_call+' import-dir --input ' + d + ' --output ' + path_to_mallet_outs + dr +'.mallet --keep-sequence')
        os.system(mallet_call+' train-topics --input ' + path_to_mallet_outs+dr+'.mallet --num-topics ' + str(num_topics) + ' --output-doc-topics ' + path_to_end_mallet_topics + dr + '-topics.txt --output-topic-keys ' +path_to_end_mallet_keys + dr +'-keys.txt --random-seed 420')
        #print(mallet_call+' train-topics --input ' + path_to_mallet_outs+dr+'.mallet --num-topics ' + str(num_topics) + ' --output-doc-topics ' + path_to_end_mallet_topics + dr + '-topics.txt --output-topic-keys ' +path_to_end_mallet_keys + dr +'-keys.txt --random-seed 420')
    with open('log_malleting.txt', 'a') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
        f.write( dt_string + ' SUCCESS')
    return 0


path_to_txts = 'git_txts/'
path_to_mallet_outs = 'mallet_outs/'
path_to_end_mallet_topics = 'topics/'
path_to_end_mallet_keys = 'keys/'
mallet_call = 'Mallet/bin/mallet'
num_topics = 30

dirs = get_directories(path_to_txts)

mallet_all_subdirs(dirs, path_to_txts, path_to_mallet_outs, path_to_end_mallet_topics, path_to_end_mallet_keys, mallet_call, num_topics)
