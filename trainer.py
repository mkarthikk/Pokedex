import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD

from sklearn.model_selection import train_test_split

from formatter import load_images
from formatter import return_label_value
from formatter import return_label_cat
from formatter import reshape_convert

all_data, all_labels, labels = load_images()
all_data = reshape_convert(all_data)
all_labels_values = return_label_value(all_labels)
all_label_cat = return_label_cat(all_labels_values)

X_train, X_test, y_train, y_test = train_test_split(all_data, all_label_cat, test_size=0.20, random_state=67)
print(X_train.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation="relu", padding="same", kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=512, activation="relu", kernel_constraint=maxnorm(3)))
model.add(Dropout(rate=0.2))
model.add(Dense(units=20, activation="softmax"))

model.compile(optimizer=SGD(lr=0.01), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x=X_train, y=y_train, epochs=200, batch_size=32)

model.save(filepath="PokeDex_Classifier.h5")


