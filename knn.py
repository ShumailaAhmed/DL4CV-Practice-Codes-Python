#OUTLINE
#Step #1 – Gather Our Dataset
#Step #2 – Split the Dataset
#Step #3 – Train the Classifier
#Step #4 - Evaluate


from sklearn.neighbors import KNeighborsClassifier#The KNeighborsClassifier is our implementation of the k-NN algorithm, provided
#by the scikit-learn library.
from sklearn.preprocessing import LabelEncoder #a helper utility to convert labels represented as strings to integers
#where there is one unique integer per class label (a common practice when applying machine
#learning)
from sklearn.model_selection import train_test_split#function used to help us create our training and testing splits
from sklearn.metrics import classification_report #evaluate the performance of our classifier
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse


# construct the argument parse and parse the arguments
ap= argparse.ArgumentParser()
ap.add_argument("-d","--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-K","--neighbors", type=int,default=1,
                help="number of nearest neighbors for cassification")
ap.add_argument("-j","--jobs", type=int,default=-1,
                help="number of jobs for k-NN distance (-1 uses all available cores)")
args= vars(ap.parse_args())
#########Step #1
# grab the list of images that we’ll be describing
print("[INFO] loading images...")
#grabs the file paths to all images in our dataset
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessor, load the dataset from disk,
#and reshape the data matrix
#initialize our SimplePreprocessor used to resize each image to 3232 pixels
sp = SimplePreprocessor(32, 32)
#The SimpleDatasetLoader is initialized, supplying our instantiated SimplePreprocessor
#as an argument (implying that sp will be applied to every image in the dataset).
sdl = SimpleDatasetLoader(preprocessors=[sp])
#cal to .load loads actual image files from disk. and return 2-tuple  each image resized to 32x32 and label
# here data is of type numpy array .shape of (3000, 32, 32,3)
#indicating there are 3,000 images in the dataset, each 32 X 32 pixels with 3 channels
(data, labels) = sdl.load(imagePaths, verbose=500)
#in order to apply the k-NN algorithm, we need to “flatten” our images from a 3D
#representation to a single list of pixel intensities
#.reshape method on the data NumPy array, flattening the 32323 images into an array with shape
#(3000, 3072)
data = data.reshape((data.shape[0], 3072))

# show some information on memory consumption of the images
#show how much memory is used by our byte array in Mb
print("[INFO] features matrix: {:.1f}MB".format(
      data.nbytes / (1024 * 1000.0)))


#############Step #2
# encode the labels as integers
#convert our labels (represented as strings) to integers where we have one
#unique integer per class. This conversion allows us to map the cat class to the integer 0, the
#dog class to integer 1, and the panda class to integer 2.
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
#Here we partition our data and labels into two unique sets: 75% of the data
#for training and 25% for testing.
#trainX=train example
#testX=test example
#trainY= train label
#testY= test lable
(trainX, testX, trainY, testY) = train_test_split(data, labels,
test_size=0.25, random_state=42)

#############Step #3
# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
#initialize the KNeighborsClassifier class
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
n_jobs=args["jobs"])
# .fit “trains” the classifier although there is no actual “learning” going on here – the k-NN
#model is simply storing the trainX and trainY data internally so it can create predictions on the
#testing set by computing the distance between the input data and the trainX data.
model.fit(trainX, trainY)

#############Step #4
#evaluate our classifier by using the classification_report function. Here
#we need to supply the testY class labels, the predicted class labels from our model, and optionally
#the names of the class labels (i.e., “dog”, “cat”, “panda”)
print(classification_report(testY, model.predict(testX),
target_names=le.classes_))
