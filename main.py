from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,Conv1D

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

# OneHot Encoding labels
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

print(x_train.shape)
print(y_train.shape)

model=Sequential()
model.add(Conv2D(132,(3,3),input_shape=(x_train.shape[1:]),activation="relu"))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(64,activation="relu"))
model.add(Dense(y_train.shape[1],activation="softmax"))
print(model.summary())

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=100,batch_size=100,validation_data=(x_test,y_test))