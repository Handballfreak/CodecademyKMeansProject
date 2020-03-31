import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans


digits = datasets.load_digits()
# print(digits.DESCR)
# print(digits.data)
# print(digits.target)


# Figure size (width, height)
fig = plt.figure(figsize=(6, 6))
# Adjust the subplots 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# For each of the 64 images
for i in range(64):
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
    # Display an image at the i-th position
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # Label the image with the target value
    ax.text(0, 7, str(digits.target[i]))
plt.show()


# Cluster Algorithm
model = KMeans(n_clusters = 10, random_state = 11)
model.fit(digits.data)

fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')
for i in range(10):
  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)
  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show()

# 0345
new_samples = np.array([
[0.00,0.00,0.00,1.07,2.29,2.29,1.15,0.00,0.00,0.00,2.90,7.48,7.02,7.02,7.55,2.21,0.00,0.92,7.17,5.27,0.00,0.00,4.43,5.57,0.00,3.97,6.03,0.08,0.00,0.00,2.29,6.10,0.00,5.12,3.82,0.00,0.00,0.00,2.52,6.10,0.00,5.34,3.05,0.00,0.00,0.00,4.35,5.34,0.00,4.81,5.80,0.23,0.00,0.69,7.32,2.60,0.00,0.92,6.79,6.64,4.73,6.26,6.19,0.08],
[0.00,0.38,1.52,1.52,0.84,0.00,0.00,0.00,0.00,3.51,7.62,7.63,7.33,1.45,0.00,0.00,0.00,0.15,0.23,0.08,5.35,5.88,0.00,0.00,0.00,0.00,3.28,5.72,7.40,5.50,0.00,0.00,0.00,0.00,2.21,5.19,7.02,5.58,0.00,0.00,0.00,0.00,0.00,0.00,1.22,7.64,0.92,0.00,0.23,4.05,3.05,3.05,4.35,7.56,1.45,0.00,0.31,5.42,6.10,6.10,5.49,2.83,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.76,3.59,0.15,0.00,0.00,0.00,0.46,4.66,7.63,6.42,0.31,0.00,0.00,2.67,7.10,6.57,1.91,6.71,5.27,0.00,1.91,7.62,7.62,7.62,7.62,7.63,7.55,5.72,0.08,1.52,1.52,1.52,1.52,4.20,7.17,2.59,0.00,0.00,0.00,0.00,0.00,0.38,2.90,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,3.14,2.29,2.28,1.37,0.00,0.00,0.00,0.00,7.63,7.17,6.86,5.04,0.00,0.00,0.00,0.00,7.63,0.76,0.00,0.00,0.00,0.00,0.00,0.00,7.63,4.19,2.22,0.00,0.00,0.00,0.00,0.00,4.88,5.57,7.56,1.60,0.00,0.00,0.00,2.59,2.75,2.29,6.11,5.81,0.00,0.00,0.00,4.12,6.72,6.87,6.87,5.72,0.00]
])

new_labels = model.predict(new_samples)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')
