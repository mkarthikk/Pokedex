import os
import numpy as np
import keras.utils as utils
from PIL import Image, ImageOps

size_32 = (32, 32)
all_data = []
all_labels = []
labels = {}


def load_images():
    count = -1
    for path, dirs, files in os.walk("C:\\Users\\muralikarthik.k\\PycharmProjects\\Pokedex\\dataset"):
        for file in files:

            # Convert each image into red, green and blue layer pixels.
            image = Image.open(os.path.join(path, file)).convert("RGB")

            # Resize the image into 32 * 32 pixels for each layer R, G and B. Apply anti alias for edges.
            img_t = ImageOps.fit(image=image, size=(32, 32), method=Image.ANTIALIAS)

            # Get the pokemon name from the path.
            label_t = path.split("\\")[-1]

            # Add it to labels dict for reference. Give a float value for each pokemon.
            if label_t not in labels.keys():
                count = count + 1
                labels.update({label_t: count})

            # Also append the label for a corresponding image into all_labels.
            all_labels.append(label_t)

            # Append the converted image to all_data.
            all_data.append(np.array(img_t))
    return np.array(all_data), np.array(all_labels), labels


def return_label_value(labels_primitive):
    all_labels_values = []
    for label_p in labels_primitive:
        for label, value in labels.items():
            if label == label_p:
                all_labels_values.append(value)
            else:
                continue
    return np.array(all_labels_values)


# Returns the labels in the form of categorical data.
# Each value in the array corresponds to probability of that label occurring.
# Useful for softmax activation function.
def return_label_cat(la_values):
    label_c = utils.to_categorical(la_values)
    return np.array(label_c)


def reshape_convert(np_arr):
    res_np_arr = []
    float_np_arrs = np_arr.astype("float32") / 255.0
    #for float_np_arr in float_np_arrs:
    #    res_np_arr.append(float_np_arr.reshape(-1))
    #return np.asarray(res_np_arr)
    return float_np_arrs

