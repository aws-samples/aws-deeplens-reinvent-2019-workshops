#!/usr/bin/env python

import requests
import os
import csv
import zipfile
# import tqdm

ZIP_FILE = './bears-for-gt-lab.zip'
ERRORS_FILE = 'download-errors.txt'
CSV_DIR = './image_csv/'
DATA_DIR = './data'

def download(url, path):
    r = requests.get(url, allow_redirects=True)
    if len(r.content) < 1024:
        raise Exception((path.split('/')[-1]).split('.')[0])
    else:
        open(path, 'wb').write(r.content)
        print('saved an image as {}'.format(path))

def main():
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)

    with zipfile.ZipFile(ZIP_FILE, 'r') as f:
        f.extractall(os.path.expanduser(CSV_DIR))

    files = list(filter(lambda x: x.endswith('csv'), os.listdir(CSV_DIR)))

    with open(ERRORS_FILE,'w') as f:
        f.write('')
    for fn in files:
        with open(CSV_DIR + fn, 'r') as f:
            reader = csv.reader(f)
            records = list(reader)[1:] # no header row

        dir_path = DATA_DIR

        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
            
        for row in records:
            path = dir_path + '/{}.jpg'.format(row[0])
            
            try:
                # If thumnail url is empty, download original url
                if not row[13]:
                    download(row[5], path)
                else:
                    download(row[13], path)
            except Exception as e:
                with open(ERRORS_FILE,'a') as f:
                    f.write(e.args[0]+'\n')

if __name__ == "__main__":
    main()



