import sys
import numpy as np

# Activation Functions
def activate(a, func="sigmoid"):

  '''a = input vector to be activated.
  func = "sigmoid"/"identity"/"tanh"/"ReLU"
  '''

  if func=="sigmoid":
    return 1/(1 + np.exp(-a))

  elif func=="identity":
    return a

  elif func=="tanh":
    return np.tanh(a)

  elif func=="ReLU":
    out = np.zeros(len(a))
    for i in range(len(a)):
      if a[i] > 0:
        out[i] = a[i]
    return out

  else:
    sys.exit("Invalid/Undefined activation function")

# Output Activation function - softmax

def softmax(a):

  '''a = input vector to softmax'''
  # Normalized inputs to softmax. Without normalization, the softmax might give nan for really high inputs
  mx = np.max(a)*np.ones(len(a)) 
  return np.exp(a-mx)/np.sum(np.exp(a-mx)) 

# Parameters (Theta) initialization functions

def theta_init(x,y,L,N,init_method="random"):

  '''
  x, y = single data input/output. 
  N = No. of neurons in each hidden layer is given by N (Assumed to be same in each hidden layer)
  L = The total no. of layers
  method = "random"/"xavier" (both are the normalized version)
  Output: theta = W, b are list of numpy arrays.
  '''

  ''' Note:
        For W, ith index connects (i-1)th hidden layer  to the ith hidden layer. 
        For b, ith index provides biases to the ith hidden layer 
        So W[1] (in common notation: W1), are the weights which connect inputs and 1st hidden layer.
        W[0] and b[0] have no significance. They are set to 0, just to maintain uniformity of indices
  '''

  N_input = x.shape[0]  # No. of neurons in the input layer or simply no. of inputs
  N_output = y.shape[0] # No. of neurons in the output layer or simply the no. of classes

  W, b = [], []
  W.append(0) # W[0] and b[0] do not play any roll. They are there just to maintain uniformity of indices
  b.append(0)

  if init_method=="random":

    # Input layer to 1st hidden layer
    W.append(np.random.randn(N,N_input)*0.01)
    b.append(np.zeros(N)) 

    # All hidden layers
    for i in range(2,L):
      W.append(np.random.randn(N,N)*0.01)
      b.append(np.zeros(N))

    # Last hidden layer to Output layer  
    W.append(np.random.randn(N_output,N)*0.01)
    b.append(np.zeros(N_output))

    return W, b

  elif init_method=="Xavier":

    # Input layer to 1st hidden layer
    sigma = (np.sqrt(6)/np.sqrt(N+N_input))
    W.append(np.random.randn(N,N_input)*sigma)
    b.append(np.zeros(N)) 

    # All hidden layers
    sigma = (np.sqrt(6)/np.sqrt(N+N))
    for i in range(2,L):
      W.append(np.random.randn(N,N)*sigma)
      b.append(np.zeros(N))

    # Last hidden layer to Output layer  
    sigma = (np.sqrt(6)/np.sqrt(N+N_output))
    W.append(np.random.randn(N_output,N)*sigma)
    b.append(np.zeros(N_output))
  
    return W, b

  else:
    sys.exit("Invalid/Undefined initialization method")


# Derivative of loss functions

def deriv_act(a, func="sigmoid"):

  if func=="sigmoid":
    g_a = activate(a,func="sigmoid")
    return np.multiply(g_a, 1-g_a)

  elif func=="identity":
    return np.ones(len(a))

  elif func=="tanh":
    g_a = activate(a,func="tanh")
    return 1 - g_a**2

  elif func=="ReLU":
    out = np.zeros(len(a))
    for i in range(len(a)):
      if a[i] > 0:
        out[i] = 1
    return out

  else:
    sys.exit("Invalid/Undefined activation function")
