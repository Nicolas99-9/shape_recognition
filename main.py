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
    print(filename)
    tar.extractall()
    tar.close()
    root = os.path.splitext(os.path.splitext(filename)[0])[0]
    # remove the .tar.gz
    print("racine :",root)
    data_folders = [os.path.join(root,d) for d in sorted(os.listdir(root))]
    if len(data_folders) != num_classes :
         raise Exception("wrond number of folders %d" %(num_classes, len(data_folders)))
    print data_folders
    return data_folders


train_folders = extract(train_filename)
test_folders = extract(test_filename)


#convert the entire dataset into a 3D array (image_index, x,y) of floating point value
# normalized the data to have zero mean and the standart deviation around 0.5 to make the training easier
#the labels will be stored into a separate arry of integers


image_size = 28
pixel_depth = 225.0 # number of levels per pixels

def load(data_folders, min_num_images, max_num_images):
  dataset = np.ndarray(
    shape=(max_num_images, image_size, image_size), dtype=np.float32)
  #create the arrays to store the images
  labels = np.ndarray(shape=(max_num_images), dtype=np.int32)
  label_index = 0
  image_index = 0
  for folder in data_folders:
    print folder
    for image in os.listdir(folder):
      if image_index >= max_num_images:
        raise Exception('More images than expected: %d >= %d' % (
          num_images, max_num_images))
      image_file = os.path.join(folder, image)
      try:
        image_data = (ndimage.imread(image_file).astype(float)- pixel_depth /2 ) / pixel_depth
        if image_data.shape != (image_size,image_size):
	     raise Exception('Unexpected image shape: %s' % str(image_data.shape))
        dataset[image_index, :,:] = image_data
        labels[image_index] = label_index
        image_index +=1
      except IOError as e:
        print 'Could not read:', image_file, ':', e, '- it\'s ok, skipping.'
    label_index += 1
  num_images = image_index
  dataset = dataset[0:num_images, :, :]
  labels = labels[0:num_images]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' % (
        num_images, min_num_images))
  print 'Full dataset tensor:', dataset.shape
  print 'Mean:', np.mean(dataset)
  print 'Standard deviation:', np.std(dataset)
  print 'Labels:', labels.shape
  return dataset, labels


train_dataset, train_labels = load(train_folders, 450000, 550000)
test_dataset, test_labels = load(test_folders, 18000, 20000)
