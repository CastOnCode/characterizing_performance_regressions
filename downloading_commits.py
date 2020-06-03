import wget
from datetime import datetime
#Open_commit_list
with open('commits_in_reverse_order.txt') as f:
    lines = f.readlines()
#delete end-of-the-line characters
for n in range(len(lines)):
    line = [char for char in lines[n][:-2]]
    k = ""
    for c in line:
        k = k+c
    lines[n] = k
#directories
dir_for_commits = "git_commits/"
dir_for_txts = "git_txts/"

#create log files
with open('log_download.txt', 'w+') as f:
    f.write('starting_to_log')
with open('log_number_of_download.txt', 'w+') as f:
    f.write('starting to log number')



run_from = 0
#running
for com_n in range(run_from, len(lines)):
    print('downloading ', com_n)
    comm = lines[com_n]
    url = 'https://github.com/git/git/archive/'+comm+'.zip'
    with open('log_download.txt', 'a') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
        f.write('\n' +dt_string+' downloading ' + url)
    saveto = dir_for_commits+comm+".zip"
    saveto = [char for char in saveto]
    k = ""
    for c in saveto:
        k = k+c
    saveto=k
    wget.download(url, saveto)
    with open('log_download.txt', 'a') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
        f.write('\n' +dt_string+' succesfully downloaded '+comm+'.zip')
    with open('log_number_of_download', 'a') as f:
        f.write('\n' +dt_string+' '+str(com_n))
    #if com_n >= 800:
     #   break


