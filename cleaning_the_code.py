import re
import os
from datetime import datetime


def get_regex():
    a = []
    a.append(r"(?<!#include )\".*\"")  #all strings but not includes
    a.append(r"[/][*](\S|\s)*?[*][/]") #all multiline comments
    a.append(r"/{2}.*")                #all single line comments
    a.append(r"[\/\.\*,\"\'\|\(\)\[\]=\+\-&\!;<>#{}]") #all punctuation
    return a

def replace_all_by_regexes(text, regexes):
    res = text
    for k in range(len(regexes)):
        reg = regexes[k]
        res = re.sub(re.compile(reg), ' ', res)
    return res


def replace_for_all_commits(regexes, dir_for_txts, dir_for_cleaned_txts):
    with open('log_cleaning.txt', 'w+') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
        f.write('\n' + dt_string + ' starting_to_log')
    dirs = get_directories(dir_for_txts)
    for k in range(len(dirs)):
        dr = dirs[k]
        com_path = dir_for_txts+dr+'/'
        clean_path = dir_for_cleaned_txts+dr+'/'
        with open('log_cleaning.txt', 'a') as f:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
            f.write('\n' + dt_string + ' cleaning_directory ' + dr)
        try:
            #print('prik1')
            #os.mkdir(clean_path)
            list_of_files = os.listdir(com_path)
            #print('prik2')
            print(list_of_files, com_path)
            for file in list_of_files:
                with open(com_path+file, mode='rt', encoding='utf-8') as f:
                    content = f.read()
                #print('OchPrik')
                content = replace_all_by_regexes(content, regexes)
                with open(clean_path+file,"w+") as f:
                    f.write(content)
            #print('prik3')
            with open('log_cleaning.txt', 'a') as f:
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
                f.write('\n' + dt_string + ' SUCCESS')
        except:
            with open('log_cleaning.txt', 'a') as f:
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
                f.write('\n' + dt_string + ' NOT SUCCESS')
        #if(k>15):
        #    return;
    return;

def get_directories(basepath):
    res = []
    for entry in os.listdir(basepath):
        if os.path.isdir(os.path.join(basepath, entry)) and entry[0]!='.':
            res.append(entry)
    return res

dir_for_txts = 'git_txts/'
dir_for_cleaned_txts = 'git_txts/'

regexes = get_regex()



replace_for_all_commits(regexes, dir_for_txts, dir_for_cleaned_txts)