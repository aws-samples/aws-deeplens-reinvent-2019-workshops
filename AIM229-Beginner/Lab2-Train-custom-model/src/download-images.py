import requests
import os
import csv
import zipfile
import time

ZIP_FILE = './open_images_bears.zip'
ERRORS_FILE = 'download-errors.txt'
CSV_DIR = './image_csv/'
DATA_DIR = './data/'

def main():
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)

    with zipfile.ZipFile(ZIP_FILE, 'r') as f:
        f.extractall(os.path.expanduser(CSV_DIR))

    files = list(filter(lambda x: x.endswith('csv'), os.listdir(CSV_DIR)))

    f = files[0]
    with open(CSV_DIR + f, 'r') as f:
        reader = csv.reader(f)
        records = list(reader)

    def download(url, path):
        r = requests.get(url, allow_redirects=True)
        if len(r.content) < 1024:
            raise Exception((path.split('/')[-1]).split('.')[0])
        else:
            open(path, 'wb').write(r.content)

    with open(ERRORS_FILE,'w') as f:
        f.write('')
    for idx,fn in enumerate(files):
        print('{}/{} {} is being processed.'.format(idx, len(files), fn))
        time.sleep(1)
        with open(CSV_DIR + fn, 'r') as f:
            reader = csv.reader(f)
            records = list(reader)[1:] # no header row
        stage = fn.split('-')[0]
        lbl = fn.split('-')[1]
        dir_path = DATA_DIR + stage
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        dir_path = DATA_DIR + '{}/{}'.format(stage,lbl)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        cnt = 0 
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