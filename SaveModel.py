import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Activation,Flatten,Conv2D,MaxPooling2D
import pickle


#Load data
X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))
#Scale
X = X/255.0

model = Sequential()
#convolve input data and impliment pooling
model.add(Conv2D(64,(3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#convolve pooled data and impliment pooling again
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#convolve pooled data and impliment pooling again
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#flatten data and setup one 64 length layer + output later
model.add(Flatten())
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])
model.fit(X,y,batch_size=32,epochs = 5, validation_split=0.1)

#save model
model.save("SomaliModel")


