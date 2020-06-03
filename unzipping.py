import os
import shutil
from zipfile import ZipFile
from datetime import datetime
import numpy as np


def get_directories(basepath):
    res = []
    for entry in os.listdir(basepath):
        if os.path.isdir(os.path.join(basepath, entry)) and entry[0]!='.':
            res.append(entry)
    return res

def copy_c_files(basepath, txtpath):
    entries = os.listdir(basepath)
    cfiles = []
    for ent in entries:
        if ent[-2:]==".c":
            cfiles.append(ent)
    for filename in cfiles:
        #print(txtpath, filename)
        dest = shutil.copyfile(basepath+filename, txtpath+filename[:-2]+'.txt')


def copy_c_with_subdirs(basepath, txtpath):
    dirs = get_directories(basepath)
    copy_c_files(basepath, txtpath)
    for d in dirs:
        copy_c_with_subdirs(basepath + d + '/', txtpath)


def extract_zip(dir_name, zip_name):
    with ZipFile(dir_name + zip_name, 'r') as z:
        os.mkdir(dir_name+zip_name[:-4])
        z.extractall(dir_name+zip_name[:-4]+'/')


def extract_all_zips_with_moving_txts(dir_name, dir_for_txts):
    with open('log_unzip.txt', 'w+') as f:
        f.write('starting_to_log')
    entries = os.listdir(dir_name)
    zipfiles = []
    i = 0
    for ent in entries:
            if ent[-4:]==".zip":
                zipfiles.append(ent)

    print(zipfiles)
    for filename in zipfiles:
        print(i)
        i = i+1
        with open('log_unzip.txt', 'a') as f:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
            f.write('\n' + dt_string + ' unzipping ' + filename)
        extract_zip(dir_name, filename)
        os.mkdir(dir_for_txts+filename[:-4]+'/')
        with open('log_unzip.txt', 'a') as f:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
            f.write('\n' + dt_string + ' successfully unzipped and created directory for txts ' + filename + ' starting to copy txts ')
        copy_c_with_subdirs(dir_name+filename[:-4]+'/', dir_for_txts+filename[:-4]+'/')
        with open('log_unzip.txt', 'a') as f:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
            f.write('\n' + dt_string + ' successfully copied c files to txts ' + filename)
        try:
            shutil.rmtree(dir_name+filename[:-4])
        except:
            print('unlucky ', filename)
            continue


dir_with_zips = 'git_commits/'
dir_with_txts = 'git_txts/'




extract_all_zips_with_moving_txts(dir_with_zips, dir_with_txts)



