from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
#make_blobs, a function used to create “blobs" of normally distributed data points
#– this is a handy function when testing or implementing our own models from scratch
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
      # compute the sigmoid activation value for a given input
      return 1.0 / (1 + np.exp(-x))
def predict(X, W):
      # take the dot product between our features and weight matrix
      preds = sigmoid_activation(X.dot(W))
      # apply a step function to threshold the outputs to binary
      # class labels
      preds[preds <= 0.5] = 0
      preds[preds > 0] = 1
      # return the predictions
      return preds
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
                help="# of epochs")
#--epochs: The number of epochs that we’ll use when training our classifier using gradient
#descent.
ap.add_argument("-a", "--alpha", type=float, default=0.01,
                help="learning rate")
#--alpha: The learning rate for the gradient descent.
args = vars(ap.parse_args())


# generate a 2-class classification problem with 1,000 data points,
# where each data point is a 2D feature vector
#call to make_blobs which generates 1,000 data points separated into
#two classes
#The labels for each of these data points are either 0 or 1
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2,
                    cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))
# insert a column of 1’s as the last entry in the feature
# matrix -- this little trick allows us to treat the bias
# as a trainable parameter within the weight matrix
#“bias trick”
X = np.c_[X, np.ones((X.shape[0]))]


# partition the data into training and testing splits using 50% of
# the data for training and the remaining 50% for testing
(trainX, testX, trainY, testY) = train_test_split(X, y,
      test_size=0.5, random_state=42)

# initialize our weight matrix and list of losses
print("[INFO] training...")
#randomly initializing our weight matrix using a uniform distribution
#such that it has the same number of dimensions as our input features
W = np.random.randn(X.shape[1], 1)
#list to keep track of our losses after each epoch
losses = []

# loop over the desired number of epochs
#start looping over the supplied number of --epochs
for epoch in np.arange(0, args["epochs"]):
      # take the dot product between our features ‘X‘ and the weight
      # matrix ‘W‘, then pass this value through our sigmoid activation
      # function, thereby giving us our predictions on the dataset
      preds = sigmoid_activation(trainX.dot(W))
      #takes the dot product between our entire training set trainX and our weight matrix
      #W. The output of this dot product is fed through the sigmoid activation function, yielding our
      #predictions.
      #the difference between our predictions and the true values
      #computes the least squares error over our predictions, a simple loss typically used for binary classification
      #problems.
      error = preds - trainY
      loss = np.sum(error ** 2)
      losses.append(loss)

      # the gradient descent update is the dot product between our
      # features and the error of the predictions
      #computing the gradient, which is the dot product between our data points X
      #and the error.
      gradient = trainX.T.dot(error)
      
      # in the update stage, all we need to do is "nudge" the weight
      # matrix in the negative direction of the gradient (hence the
      # term "gradient descent" by taking a small step towards a set
      # of "more optimal" parameters
      W += -args["alpha"] * gradient

      # check to see if an update should be displayed
      if epoch == 0 or (epoch + 1) % 5 == 0:
            print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1),
                  loss))
            
# evaluate our model
print("[INFO] evaluating...")
#To actually make predictions using our weight matrix W, we call the predict method on testX
#and W
preds = predict(testX, W)
#Given the predictions, we display a nicely formatted classification report to our
#terminal
print(classification_report(testY, preds))

#plotting (1) the testing data so we can visualize the dataset we are
#trying to classify and (2) our loss over time

# plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY, s=30)

# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()


