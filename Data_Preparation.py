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

  # Normalizing the data
  x_train, x_test = x_train/255, x_test/255

  # Randomly shuffle train data then split it into train and validatation data
  shuffle_indices = np.random.permutation(range(len(y_train))) # Creating random indices for shuffling
  x_train, y_train = x_train[shuffle_indices,:,:], y_train[shuffle_indices] # Shuffled train data

  val_fraction = 0.1 # 10% of train data to be used for validation
  len_train = int((1-val_fraction)*len(y_train))  # Length of train data after removing validation from it
  x_train, x_val = x_train[:len_train], x_train[len_train:]
  y_train, y_val = y_train[:len_train], y_train[len_train:]

  return x_train, x_test, x_val, y_train, y_test, y_val, labels
