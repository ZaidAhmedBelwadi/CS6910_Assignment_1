from data_preparation import data # Importing 'data_preparation.py' file
import matplotlib.pyplot as plt
import numpy as np

x_train, x_test, x_val, y_train, y_test, y_val, labels = data()

# Creating Sample Set that contains all the classes.
sample_x = []
sample_label = []
i = 0

while len(sample_label)<10:
  if labels[list(y_train[i]).index(1)] not in sample_label:
    sample_x.append(x_train[i])
    sample_label.append(labels[list(y_train[i]).index(1)])
  i+=1

# Reshaping the sample data back into 28x28
sample_x = np.array(sample_x)
sample_x = sample_x.reshape(10,28,28)

# Plotting sample set using imshow
fig = plt.figure(figsize=[13,6])
for k in range(1,11):
  plt.subplot(2, 5, k)
  plt.imshow(sample_x[k-1], cmap=plt.get_cmap('hot'))
  ax = plt.gca()
  ax.set_title(sample_label[k-1])
  ax.axes.xaxis.set_visible(False)
  ax.axes.yaxis.set_visible(False)

plt.show()