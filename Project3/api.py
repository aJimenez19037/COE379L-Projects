import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from flask import Flask, request
import numpy as np

#laod in model
alt_lenet5_cnn = tf.keras.models.load_model('keras_models/alt_lenet5.keras')
lenet5_cnn = tf.keras.models.load_model('keras_models/lenet5.keras')

# # pre-processing...
# # normalize
# test_data_dir = 'images-cnn-split/test/'
# batch_size = 2
# img_height = 128
# img_width = 128
# test_ds = tf.keras.utils.image_dataset_from_directory(
# test_data_dir,
# seed=123,
# image_size=(img_height, img_width),
# )

# # approach 1: manually rescale data --
# rescale = Rescaling(scale=1.0/255)
# test_rescale_ds = test_ds.map(lambda image,label:(rescale(image),label))

# # test_loss, test_accuracy = alt_lenet5_cnn.evaluate(test_rescale_ds, verbose=0)
# # test_accuracy


app = Flask(__name__)

def preprocess_input(im):
    """
    Converts user-provided input into an array that can be used with the model.
    This function could raise an exception.
    """
    # Convert image to numpy array
    d = np.array(im)
    d = d/255
    # Add an extra dimension to simulate batch dimension
    d = np.expand_dims(d, axis=0)
    return d
@app.route('/models/alt_lenet5', methods=['GET'])
def model_info_alt_lenet5():
   return {
      "version": "v1",
      "name": "alt_lenet5",
      "description": "Classify hurricane images into damage or not damage housing",
      "number_of_parameters": 3453121
   }
@app.route('/models/alt_lenet5', methods=['POST'])
def classify_alt_lenet5():
   im = request.json.get('image')
   if not im:
      return {"error": "The `image` field is required"}, 404
   try:
      data = preprocess_input(im)
   except Exception as e:
      return {"error": f"Could not process the `image` field; details: {e}"}, 404
   return { "result": alt_lenet5_cnn.predict(data).tolist()}

@app.route('/models/lenet5', methods=['GET'])
def model_info_lenet5():
   return {
      "version": "v1",
      "name": "lenet5",
      "description": "Classify hurricane images into damage or not damage housing based on lenet model",
      "number_of_parameters": 857458
   }
@app.route('/models/lenet5', methods=['POST'])
def classify_lenet5():
   im = request.json.get('image')
   if not im:
      return {"error": "The `image` field is required"}, 404
   try:
      data = preprocess_input(im)
   except Exception as e:
      return {"error": f"Could not process the `image` field; details: {e}"}, 404
   return { "result": lenet5_cnn.predict(data).tolist()}

# start the development server
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')




