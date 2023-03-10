from func_defs import activate,softmax,deriv_act
import numpy as np
import sys

def forward_pass(x, theta, L, activation):
  
  '''
  x = single data input
  theta = parameters (W,b) 
  L = total no. of layers => L = nhl - 1 (nhl = number of hidden layers)
  activation="sigmoid"/"identity"/"tanh"/"ReLU"
  '''
  '''
  Note that the indices of a, W, b begin from 1. So a[1] refers to pre-activation of first hidden layer
  W[1] refers to the weight matrix connecting 1st hidden layer and input neurons
  a[0], W[0], b[0] have no significance. They are set to 0, just to maintain uniformity of indices
  Only h[0] is meaningful and is the input vector
  '''

  # Retrieving weights and biases from theta
  W, b = theta

  # Initialize a's and h's. Stored as a list which has ndarrays as its elements.
  a = []
  h = []
  h.append(x) # h[0] set to x
  a.append(0) # a[0] has no significance. It is set as 0 just to maintain uniformity of indices

  # Forward pass till L-1 layers
  for k in range(1,L):
    a.append(b[k] + np.matmul(W[k],h[k-1]))
    h.append(activate(a[k],func=activation))

  # Final pass
  a.append(b[L] + np.matmul(W[L],h[L-1]))
  y_hat = softmax(a[L])
  h.append(y_hat)

  return a, h, y_hat


def back_prop(y, theta, L, a, h, y_hat, loss, activation):

  '''
  y = single output data (true class)
  theta = parameters (W,b)
  L = Total No. of layers
  a = pre-activation
  h = activation
  y_hat = predicted y
  loss="cross_entropy"/"mean_squared_error"
  activation="sigmoid"/"identity"/"tanh"/"ReLU"
  '''

  '''
  All dels are stored as lists where its index i represents the gradient in ith layer.
  Again del_a[0],del_h[0],del_w[0],del_b[0] have no significance whatsover and is just set as zero.
  '''
  W, b = theta
  del_a, del_h, del_W, del_b = [0]*(L+1), [0]*(L+1), [0]*(L+1), [0]*(L+1)

  # At output layer
  if loss=="cross_entropy":
    del_a[L] = -(y-y_hat)
  elif loss=="mean_squared_error":
    A = np.sum(y_hat**2) - np.dot(y, y_hat)
    del_a[L] = -2*(np.multiply(y_hat, (A + y_hat - y)))
  else:
    sys.exit("Invalid/Undefined loss function")

  # Hidden layers
  for k in range(L,1,-1):
    del_W[k] = np.outer(del_a[k],h[k-1])
    del_b[k] = del_a[k]
    del_h[k-1] = np.matmul(W[k].T,del_a[k])
    del_a[k-1] = np.multiply(del_h[k-1],deriv_act(a[k-1],func=activation))

  # At input layer
  del_W[1] = np.outer(del_a[1],h[0])
  del_b[1] = del_a[1]

  return del_W, del_b 


# Total loss and accuracy

def metrics(x_data, y_data, theta, L, activation, alpha, func="cross_entropy"):

  y_hat_data = np.zeros(y_data.shape)
  loss = 0

  # Forward propogation with data
  for i in range(len(y_data)):
    y_hat_data[i] = forward_pass(x_data[i], theta, L, activation)[2]


  # Total loss

  if func=="cross_entropy":
    N_samples = y_data.shape[0]
    for k in range(N_samples):
      loss+= -np.log(y_hat_data[k][list(y_data[k]).index(1)])/(N_samples)

  elif func=="mean_squared_error":

    N_samples = y_data.shape[0] 
    for k in range(N_samples):
      loss+= np.sum((y_hat_data[k] - y_data[k])**2)/(N_samples)

  else:
    sys.exit("Invalid/Undefined loss function")

  # Adding Regularization term
  W = theta[1]
  for k in range(1, L+1):
    loss+= (alpha/(2*N_samples))*np.sum(W[k]**2)

  # Accuracy
  corr_pred = 0
  for i in range(len(y_data)):
    corr_pred += y_data[i][np.argmax(y_hat_data[i])]
  accuracy = str(100*corr_pred/(len(y_data)))+"%"

  # Returning all the metrics
  return loss, accuracy