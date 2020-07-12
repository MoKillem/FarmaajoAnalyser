import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

#directory of image
DATADIR = r"C:\Users\codew\OneDrive\Documents\projects\SomaliChecker\DataSets\images"
CATEGORIES = ["Somali"]
#create training data
training_data = []
size_image = 150
#create training data
def create_training_data():
    for category in CATEGORIES:
        # get path of dog and cat
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        # loop through path
        for img in os.listdir(path):
            try:
                # read the images
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                # resize img_array
                new_array = cv2.resize(img_array, (size_image, size_image))
                # append data
                training_data.append([new_array, class_num])
                print('loading')
            except Exception as e:
                pass
create_training_data()
#Shuffle Training Data
random.shuffle(training_data)

#split labels and actual training data
X = []
y = []
for feat,label in training_data:
    X.append(feat)
    y.append(label)
# convert X into np array and reshape to 50 by 50 by 1
X = np.array(X).reshape(-1, size_image, size_image, 1)
y = np.array(y)
#save looped datasets
pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

plt.imshow(training_data[101][0], cmap="gray")
plt.show()
