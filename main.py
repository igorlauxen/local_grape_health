from keras.datasets import mnist
import matplotlib.pyplot as plt

#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#plot the first image in the dataset
plt.imshow(X_train[0])

IMAGE_SIZE = 32

#normalizing the dataset
train_X, val_X = train_X/IMAGE_SIZE, val_X/IMAGE_SIZE