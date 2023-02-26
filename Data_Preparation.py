from keras.datasets import fashion_mnist, mnist
import numpy as np

def data_preparation(dataset="fashion_mnist"):

  # Choosing dataset
  if dataset == "fashion_mnist":
    labels = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  elif dataset == "mnist":
    labels = ['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine']
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # Merging the standard train/test split 
  x_data, y_data = np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0)
  
  # Randomly shuffle data then split train, test, validatation data (along with normalization)
  shuffle_indices = np.random.permutation(range(len(y_data)))
  x_data, y_data = x_data[shuffle_indices,:,:], y_data[shuffle_indices]
  x_train, x_test, x_val = x_data[:54000]/255, x_data[54000:60000]/255, x_data[60000:]/255
  y_train, y_test, y_val = y_data[:54000], y_data[54000:60000], y_data[60000:]

  return x_train, x_test, x_val, y_train, y_test, y_val, labels