from forw_back_prop import forward_pass, back_prop
import numpy as np

# Stochastic gradient descent

def sgd_one_epoch(x_train, y_train, theta, L, activation, loss, eta, alpha):

# Training for single epoch
  W, b = theta
  # dW,db = single data point grads
  del_W, del_b = [0]+[np.zeros(W[k].shape) for k in range(1,L+1)], [0]+[np.zeros(b[k].shape) for k in range(1,L+1)]

  for i in range(len(y_train)):
    a, h, y_hat = forward_pass(x_train[i], theta, L, activation)
    dW, db = back_prop(y_train[i], theta, L, a, h, y_hat, loss, activation, alpha)
    W[1:] = [W[k] - eta*dW[k] for k in range(1,L+1)]
    b[1:] = [b[k] - eta*db[k] for k in range(1,L+1)]
    theta = (W, b)
  return theta


# Momentum based gradient descent

def momentum_one_epoch(optimizer, x_train, y_train, theta, L, activation, loss, eta, alpha, batch_size, momentum,
                       theta_momentum = ()):

# Training for single epoch
  W, b = theta
  W_momentum, b_momentum = theta_momentum
  
  # dW,db = single data point grads. del_W, del_b = Accumulation of grads
  del_W, del_b = [0]+[np.zeros(W[k].shape) for k in range(1,L+1)], [0]+[np.zeros(b[k].shape) for k in range(1,L+1)]
  count = 0

  for i in range(len(y_train)):
    
    a, h, y_hat = forward_pass(x_train[i], theta, L, activation)
    dW, db = back_prop(y_train[i], theta, L, a, h, y_hat, loss, activation, alpha)
    del_W[1:] = [del_W[k]+dW[k] for k in range(1,L+1)]
    del_b[1:] = [del_b[k]+db[k] for k in range(1,L+1)]
    count +=1

    if count==batch_size:
      W_momentum, b_momentum = theta_momentum
      W_update = [0]+[momentum*W_momentum[k] + del_W[k] for k in range(1,L+1)]
      b_update = [0]+[momentum*b_momentum[k] + del_b[k] for k in range(1,L+1)]
      W[1:] = [W[k] - eta*W_update[k] for k in range(1,L+1)]
      b[1:] = [b[k] - eta*b_update[k] for k in range(1,L+1)] 
      
      theta = (W, b)
      theta_momentum = (W_update, b_update)
      
      del_W, del_b = [0]+[np.zeros(W[k].shape) for k in range(1,L+1)], [0]+[np.zeros(b[k].shape) for k in range(1,L+1)]
      count=0
  
  return theta, theta_momentum


# Nesterov accelerated gradient descent

def nag_one_epoch(optimizer, x_train, y_train, theta, L, activation, loss, eta, alpha, batch_size, momentum,
                  theta_momentum = ()):
  
# Training for single epoch
  W, b = theta
  W_momentum, b_momentum = theta_momentum
  W_lookahead = [0]+[W[k] - eta*momentum*W_momentum[k] for k in range(1,L+1)]
  b_lookahead = [0]+[b[k] - eta*momentum*b_momentum[k] for k in range(1,L+1)]
  theta_lookahead = (W_lookahead, b_lookahead)

  # dW,db = single data point grads. del_W, del_b = Accumulation of grads
  del_W, del_b = [0]+[np.zeros(W[k].shape) for k in range(1,L+1)], [0]+[np.zeros(b[k].shape) for k in range(1,L+1)]
  count = 0

  for i in range(len(y_train)):

    a, h, y_hat = forward_pass(x_train[i], theta_lookahead, L, activation)
    dW, db = back_prop(y_train[i], theta_lookahead, L, a, h, y_hat, loss, activation, alpha)
    del_W[1:] = [del_W[k]+dW[k] for k in range(1,L+1)]
    del_b[1:] = [del_b[k]+db[k] for k in range(1,L+1)]
    count +=1

    if count==batch_size:
      W_momentum, b_momentum = theta_momentum
      W_update = [0]+[momentum*W_momentum[k] + del_W[k] for k in range(1,L+1)]
      b_update = [0]+[momentum*b_momentum[k] + del_b[k] for k in range(1,L+1)]
      W[1:] = [W[k] - eta*W_update[k] for k in range(1,L+1)]
      b[1:] = [b[k] - eta*b_update[k] for k in range(1,L+1)] 
      
      theta = (W, b)
      theta_momentum = (W_update, b_update)
      W_lookahead[1:] = [W[k] - eta*momentum*W_update[k] for k in range(1,L+1)]
      b_lookahead[1:] = [b[k] - eta*momentum*b_update[k] for k in range(1,L+1)]
      theta_lookahead = (W_lookahead, b_lookahead)
      
      del_W, del_b = [0]+[np.zeros(W[k].shape) for k in range(1,L+1)], [0]+[np.zeros(b[k].shape) for k in range(1,L+1)]
      count=0
  
  return theta, theta_momentum


# RMSProp

def rmsprop_one_epoch(optimizer, x_train, y_train, theta, L, activation, loss, eta, alpha, batch_size,
                      beta,epsilon, v_theta_history=()):
  
# Training for single epoch
  W, b = theta

  # dW,db = single data point grads. del_W, del_b = Accumulation of grads
  del_W, del_b = [0]+[np.zeros(W[k].shape) for k in range(1,L+1)], [0]+[np.zeros(b[k].shape) for k in range(1,L+1)]
  count = 0

  for i in range(len(y_train)):

    a, h, y_hat = forward_pass(x_train[i], theta, L, activation)
    dW, db = back_prop(y_train[i], theta, L, a, h, y_hat, loss, activation, alpha)
    del_W[1:] = [del_W[k]+dW[k] for k in range(1,L+1)]
    del_b[1:] = [del_b[k]+db[k] for k in range(1,L+1)]
    count +=1

    if count==batch_size:
      v_W_history, v_b_history = v_theta_history
      v_W_curr = [0]+[beta*v_W_history[k] + (1-beta)*np.sum(del_W[k]**2) for k in range(1,L+1)] 
      v_b_curr = [0]+[beta*v_b_history[k] + (1-beta)*np.sum(del_b[k]**2) for k in range(1,L+1)]
 
      W[1:] = [W[k] - (eta/(np.sqrt(v_W_curr[k])+epsilon))*del_W[k] for k in range(1,L+1)]
      b[1:] = [b[k] - (eta/(np.sqrt(v_W_curr[k])+epsilon))*del_b[k] for k in range(1,L+1)] 

      theta = (W, b)
      v_theta_history = (v_W_curr, v_b_curr)
      del_W, del_b = [0]+[np.zeros(W[k].shape) for k in range(1,L+1)], [0]+[np.zeros(b[k].shape) for k in range(1,L+1)]
      count=0

  return theta, v_theta_history
  
  
# Adam

def adam_one_epoch(optimizer, x_train, y_train, theta, L, activation, loss, eta, alpha, batch_size,
                      beta1, beta2, epsilon, v_theta_history=(), theta_momentum_prev=(), update_count=1):
  
# Training for single epoch
  W, b = theta
  W_momentum_prev, b_momentum_prev = theta_momentum_prev

  # dW,db = single data point grads. del_W, del_b = Accumulation of grads
  del_W, del_b = [0]+[np.zeros(W[k].shape) for k in range(1,L+1)], [0]+[np.zeros(b[k].shape) for k in range(1,L+1)]
  count = 0

  for i in range(len(y_train)):

    a, h, y_hat = forward_pass(x_train[i], theta, L, activation)
    dW, db = back_prop(y_train[i], theta, L, a, h, y_hat, loss, activation, alpha)
    del_W[1:] = [del_W[k]+dW[k] for k in range(1,L+1)]
    del_b[1:] = [del_b[k]+db[k] for k in range(1,L+1)]
    count +=1

    if count==batch_size:
      W_momentum_prev, b_momentum_prev = theta_momentum_prev
      W_momentum_curr = [0]+[beta1*W_momentum_prev[k] + (1-beta1)*del_W[k] for k in range(1,L+1)]
      b_momentum_curr = [0]+[beta1*b_momentum_prev[k] + (1-beta1)*del_b[k] for k in range(1,L+1)]
      W_momentum_curr_hat = [0]+[W_momentum_curr[k]/(1 - beta1**update_count) for k in range(1,L+1)]
      b_momentum_curr_hat = [0]+[b_momentum_curr[k]/(1 - beta1**update_count) for k in range(1,L+1)]

      v_W_history, v_b_history = v_theta_history
      v_W_curr = [0]+[beta2*v_W_history[k] + (1-beta2)*np.sum(del_W[k]**2) for k in range(1,L+1)] 
      v_b_curr = [0]+[beta2*v_b_history[k] + (1-beta2)*np.sum(del_b[k]**2) for k in range(1,L+1)]
      v_W_curr_hat = [0]+[v_W_curr[k]/(1 - beta2**update_count) for k in range(1,L+1)]
      v_b_curr_hat = [0]+[v_b_curr[k]/(1 - beta2**update_count) for k in range(1,L+1)]

      W[1:] = [W[k] - (eta/(np.sqrt(v_W_curr_hat[k])+epsilon))*W_momentum_curr_hat[k] for k in range(1,L+1)]
      b[1:] = [b[k] - (eta/(np.sqrt(v_b_curr_hat[k])+epsilon))*b_momentum_curr_hat[k] for k in range(1,L+1)] 

      theta = (W, b)
      v_theta_history = (v_W_curr, v_b_curr)
      theta_momentum_prev = (W_momentum_curr, b_momentum_curr)
      update_count+=1

      del_W, del_b = [0]+[np.zeros(W[k].shape) for k in range(1,L+1)], [0]+[np.zeros(b[k].shape) for k in range(1,L+1)]
      count=0

  return theta, v_theta_history, theta_momentum_prev, update_count


# NAdam

def nadam_one_epoch(optimizer, x_train, y_train, theta, L, activation, loss, eta, alpha, batch_size,
                      beta1, beta2, epsilon, v_theta_history=(), theta_momentum_prev=(), update_count=1):
  
# Training for single epoch
  W, b = theta
  W_momentum_prev, b_momentum_prev = theta_momentum_prev

  # dW,db = single data point grads. del_W, del_b = Accumulation of grads
  del_W, del_b = [0]+[np.zeros(W[k].shape) for k in range(1,L+1)], [0]+[np.zeros(b[k].shape) for k in range(1,L+1)]
  count = 0

  for i in range(len(y_train)):

    a, h, y_hat = forward_pass(x_train[i], theta, L, activation)
    dW, db = back_prop(y_train[i], theta, L, a, h, y_hat, loss, activation, alpha)
    del_W[1:] = [del_W[k]+dW[k] for k in range(1,L+1)]
    del_b[1:] = [del_b[k]+db[k] for k in range(1,L+1)]
    count +=1

    if count==batch_size:
      W_momentum_prev, b_momentum_prev = theta_momentum_prev
      W_momentum_curr = [0]+[beta1*W_momentum_prev[k] + (1-beta1)*del_W[k] for k in range(1,L+1)]
      b_momentum_curr = [0]+[beta1*b_momentum_prev[k] + (1-beta1)*del_b[k] for k in range(1,L+1)]
      W_momentum_curr_hat = [0]+[W_momentum_curr[k]/(1 - beta1**update_count) for k in range(1,L+1)]
      b_momentum_curr_hat = [0]+[b_momentum_curr[k]/(1 - beta1**update_count) for k in range(1,L+1)]

      v_W_history, v_b_history = v_theta_history
      v_W_curr = [0]+[beta2*v_W_history[k] + (1-beta2)*np.sum(del_W[k]**2) for k in range(1,L+1)] 
      v_b_curr = [0]+[beta2*v_b_history[k] + (1-beta2)*np.sum(del_b[k]**2) for k in range(1,L+1)]
      v_W_curr_hat = [0]+[v_W_curr[k]/(1 - beta2**update_count) for k in range(1,L+1)]
      v_b_curr_hat = [0]+[v_b_curr[k]/(1 - beta2**update_count) for k in range(1,L+1)]

      W[1:] = [W[k] - (eta/(np.sqrt(v_W_curr_hat[k])+epsilon))*(beta1*W_momentum_curr_hat[k] + ((1-beta1)/(1 - beta1**update_count))*del_W[k]) for k in range(1,L+1)]
      b[1:] = [b[k] - (eta/(np.sqrt(v_b_curr_hat[k])+epsilon))*(beta1*b_momentum_curr_hat[k] + ((1-beta1)/(1 - beta1**update_count))*del_b[k]) for k in range(1,L+1)] 

      theta = (W, b)
      v_theta_history = (v_W_curr, v_b_curr)
      theta_momentum_prev = (W_momentum_curr, b_momentum_curr)
      update_count+=1

      del_W, del_b = [0]+[np.zeros(W[k].shape) for k in range(1,L+1)], [0]+[np.zeros(b[k].shape) for k in range(1,L+1)]
      count=0

  return theta, v_theta_history, theta_momentum_prev, update_count