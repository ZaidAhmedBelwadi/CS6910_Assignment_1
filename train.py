import numpy as np
from data_preparation import data
from func_defs import theta_init
from optimizers import sgd_one_epoch, momentum_one_epoch, nag_one_epoch, rmsprop_one_epoch, adam_one_epoch, nadam_one_epoch 
from forw_back_prop import metrics


def train_NN(epochs=1, batch_size=4, loss="cross_entropy", optimizer="sgd", 
             learning_rate=0.1, momentum=0.5, beta=0.5, beta1=0.5, beta2=0.5,
             epsilon=0.000001, weight_decay=0.0, weight_init="random",
             num_layers=1, hidden_size=4, activation="sigmoid", dataset="fashion_mnist"):

  # Get data
  x_train, x_test, x_val, y_train, y_test, y_val, labels = data(dataset)

  L = num_layers+1  # Total no. of layers
  loss_train = []
  loss_val = []
  val_accuracy = []
  y_hat_train = np.zeros(y_train.shape)
  y_hat_val = np.zeros(y_val.shape)

  # Initialize parameters
  theta = theta_init(x_train[0], y_train[0], L, hidden_size, init_method=weight_init)
  W,b = theta

  # Initial Training and Validation loss
  loss_train.append(metrics(x_train, y_train, theta, L, activation, weight_decay, func = loss)[0])
  curr_loss_val, curr_val_accuracy = metrics(x_val, y_val, theta, L, activation, weight_decay, func = loss)
  loss_val.append(curr_loss_val)
  val_accuracy.append(curr_val_accuracy)
  print("At 0th iteration:")
  print("\tTrain loss =", loss_train[-1])
  print("\tValidation loss =", curr_loss_val)
  print("\tValidation Accuracy =", curr_val_accuracy)


  # Training network using various optimizers

  for epoch_no in range(1,epochs+1):
    
    if optimizer == "sgd":
      theta = sgd_one_epoch(x_train, y_train, theta, L, activation, loss, learning_rate, weight_decay, batch_size)

    elif optimizer == "momentum":
      if epoch_no == 1:
        theta_momentum = ([0]+[np.zeros(W[k].shape) for k in range(1,L+1)], [0]+[np.zeros(b[k].shape) for k in range(1,L+1)])

      theta, theta_momentum = momentum_one_epoch(x_train, y_train, theta, L, activation, loss, learning_rate, weight_decay, batch_size, momentum, 
                                                 theta_momentum=theta_momentum)
    elif optimizer == "nag":
      if epoch_no == 1:
        theta_momentum = ([0]+[np.zeros(W[k].shape) for k in range(1,L+1)], [0]+[np.zeros(b[k].shape) for k in range(1,L+1)])

      theta, theta_momentum = nag_one_epoch(x_train, y_train, theta, L, activation, loss, learning_rate, weight_decay, batch_size, momentum,
                                            theta_momentum=theta_momentum) 
      
    elif optimizer == "rmsprop":
      if epoch_no == 1:
        v_theta_history = ([0]*(L+1), [0]*(L+1))

      theta, v_theta_history = rmsprop_one_epoch(x_train, y_train, theta, L, activation, loss, learning_rate, weight_decay, batch_size, 
                                                 beta, epsilon, v_theta_history=v_theta_history) 

    elif optimizer == "adam":
      if epoch_no == 1:
        v_theta_history = ([0]*(L+1), [0]*(L+1))
        theta_momentum_prev = ([0]+[np.zeros(W[k].shape) for k in range(1,L+1)], [0]+[np.zeros(b[k].shape) for k in range(1,L+1)])
        update_count = 1

      theta, v_theta_history, theta_momentum_prev, update_count = adam_one_epoch(x_train, y_train, theta, L, activation, loss, learning_rate, weight_decay,
                                                                                 batch_size, beta1, beta2, epsilon, v_theta_history=v_theta_history,
                                                                                 theta_momentum_prev=theta_momentum_prev, update_count=update_count)

    elif optimizer == "nadam":
      if epoch_no == 1:
        v_theta_history = ([0]*(L+1), [0]*(L+1))
        theta_momentum_prev = ([0]+[np.zeros(W[k].shape) for k in range(1,L+1)], [0]+[np.zeros(b[k].shape) for k in range(1,L+1)])
        update_count = 1

      theta, v_theta_history, theta_momentum_prev, update_count = nadam_one_epoch(x_train, y_train, theta, L, activation, loss, learning_rate, weight_decay,
                                                                                 batch_size, beta1, beta2, epsilon, v_theta_history=v_theta_history,
                                                                                 theta_momentum_prev=theta_momentum_prev, update_count=update_count)
    

    # Calculating train and validation loss
    loss_train.append(metrics(x_train, y_train, theta, L, activation, weight_decay, func = loss)[0])
    curr_loss_val, curr_val_accuracy = metrics(x_val, y_val, theta, L, activation, weight_decay, func = loss)
    loss_val.append(curr_loss_val)
    val_accuracy.append(curr_val_accuracy)
    print("\nAfter", epoch_no, "iteration(s):")
    print("\tTrain loss =", loss_train[-1])
    print("\tValidation loss =", curr_loss_val)
    print("\tValidation Accuracy =", curr_val_accuracy)
