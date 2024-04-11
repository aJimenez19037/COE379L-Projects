import requests
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from PIL import Image
import numpy as np
import sys

def main(argv):
    file_name = argv[0]
    img = Image.open(file_name)
    img = np.asarray(img)
    rsp = requests.post("http://172.17.0.1:5000/models/alt_lenet5", json={"image": img.tolist()})
    print(rsp.json())

if __name__ == "__main__":
        main(sys.argv[1:])
