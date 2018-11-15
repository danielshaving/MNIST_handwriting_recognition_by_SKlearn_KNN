# K-Nearest Neighbor Classification

from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2

# load the MNIST digits dataset
mnist = datasets.load_digits()


# Training and testing split,
# 75% for training and 25% for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data), mnist.target, test_size=0.25, random_state=42)

# take 10% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1, random_state=84)

# Checking sizes of each data split
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))


# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k
kVals = range(1, 30, 2)
accuracies = []

# loop over kVals
for k in range(1, 30, 2):
    # train the classifier with the current value of `k`
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabels)

    # evaluate the model and print the accuracies list
    score = model.score(valData, valLabels)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)

# largest accuracy
# np.argmax returns the indices of the maximum values along an axis
i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
    accuracies[i] * 100))


# Now that I know the best value of k, re-train the classifier
model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(trainData, trainLabels)



# Predict labels for the test set
predictions = model.predict(testData)

# Evaluate performance of model for each of the digits
print("EVALUATION ON TESTING DATA")
print(classification_report(testLabels, predictions))

# some indices are classified correctly 100% of the time (precision = 1)
# high accuracy (98%)

# check predictions against images
# loop over a few random digits
image = testData
j = 0
for i in np.random.randint(0, high=len(testLabels), size=(24,)):
        # np.random.randint(low, high=None, size=None, dtype='l')
    prediction = model.predict(image)[i]
    image0 = image[i].reshape((8, 8)).astype("uint8")
    image0 = exposure.rescale_intensity(image0, out_range=(0, 255))
    plt.subplot(4,6,j+1)
    plt.title(str(prediction))
    plt.imshow(image0,cmap='gray')
    plt.axis('off')


        # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
        # then resize it to 32 x 32 pixels for better visualization

        #image0 = imutils.resize(image[0], width=32, inter=cv2.INTER_CUBIC)

    j = j+1

    # show the prediction
    # print("I think that digit is: {}".format(prediction))
    # print('image0 is ',image0)
    # cv2.imshow("Image", image0)
    # cv2.waitKey(0) # press enter to view each one!
plt.show()
