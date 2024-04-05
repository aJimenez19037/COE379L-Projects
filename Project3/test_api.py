import requests
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


# NOTE: we need to perform the same pre-processing...
# normalize
test_data_dir = 'images-cnn-split/test/'
batch_size = 2
# this is what was used in the paper --
img_height = 128
img_width = 128
# note that subset="training", "validation", "both", and dictates what is returned
test_ds = tf.keras.utils.image_dataset_from_directory(
test_data_dir,
seed=123,
image_size=(img_height, img_width),
)
# grab an entry from X_test -- here, we grab the first one
l = X_test[0].tolist()

# make the POST request passing the sinlge test case as the `image` field:
rsp = requests.post("http://172.17.0.1:5000/models/clothes/v1", json={"image": l})

# print the json response
rsp.json()