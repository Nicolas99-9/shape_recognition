import matplotlib.pyplot as plt
import numpy as np
import os
import tarfile
import urllib
from IPython.display import display
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
import cPickle as pickle

#load the data to the local machine
#the data is a set of characters rendered in a variety of fonts of 28*28 pixels
#the label are limited to 'A' through 'J'



url = 'http://yaroslavvb.com/upload/notMNIST/'

def download(filename, excepeted_bytes):
    if not os.path.exists(filename):
        filename,_ = urllib.urlretrieve(url+filename, filename)
        #get the url of the file to load and check if it's possible to download it
    statinfo  = os.stat(filename)
    if statinfo.st_size == excepeted_bytes:
        print 'File found', filename
    else:
        raise Exception('Failed to verify' + filename)
    return filename

train_filename = download('notMNIST_large.tar.gz', 247336696)
test_filename = download('notMNIST_small.tar.gz',    8458043)



#extract the data and give a set of directories labelled from A to J

num_classes = 10

def extract(filename):
    tar = tarfile.open(filename)
    tar.extractall()
    tar.close()
    root = os.path.splitext(os.path.splitext(filename)[0])[0]
    # remove the .tar.gz
    data_folders = [os.path.join(root,d) for d in sorted(os.listdir(root))]
    if len(data_folders) != num_classes :
         raise Exception("wrond number of folders %d" %(num_classes, len(data_folders)))
    print data_folders
    return data_folders

train_folders = extract(train_filename)
test_folders = extract(test_filename)

