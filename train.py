from func_defs import activate,softmax,theta_init
import numpy as np

def forward_pass(x, theta, L, activation="sigmoid"):
  
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

x = np.array([0.5,0.2,0.33,1,0,0.7])
y = np.array([1,0])
L, N = 5, 10
theta = theta_init(x,y,5,10,init_method="xavier")
a, h, y_hat = forward_pass(x, theta, L, activation="sigmoid")
print(y_hat)
