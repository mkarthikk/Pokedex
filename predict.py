from sklearn.model_selection import train_test_split
from keras.models import load_model
import numpy as np
from PIL import Image
from formatter import load_images
from formatter import return_label_value
from formatter import return_label_cat
from formatter import reshape_convert

all_data, all_labels, labels = load_images()
all_data = reshape_convert(all_data)
all_labels_values = return_label_value(all_labels)
print(labels)
all_label_cat = return_label_cat(all_labels_values)

X_train, X_test, y_train, y_test = train_test_split(all_data, all_label_cat, test_size=0.20, random_state=67)

model = load_model("PokeDex_Classifier.h5")
#results = model.evaluate(x=X_test, y=y_test)
hello = X_test[45]
img = Image.fromarray((hello * 255).astype(np.uint8))
img.show()
hello = hello.reshape(1, 32, 32, 3)

hello1 = model.predict(x=hello)
print(np.argmax(hello1[0]))
