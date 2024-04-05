import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from flask import Flask

#laod in model
alt_lenet5_cnn = tf.keras.models.load_model('alt_lenet5.keras')

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
   # convert to a numpy array
   d = np.array(im)
   # then add an extra dimension
   return d.reshape(3, 128, 128)
@app.route('/models/alt_lenet5', methods=['GET'])
def model_info():
   return {
      "version": "v1",
      "name": "alt_lenet5",
      "description": "Classify hurricane images into damage or not damage housing",
      "number_of_parameters": 3453121
   }
@app.route('/models/alt_lenet5', methods=['POST'])
def classify_clothes_image():
   im = request.json.get('image')
   if not im:
      return {"error": "The `image` field is required"}, 404
   try:
      data = preprocess_input(im)
   except Exception as e:
      return {"error": f"Could not process the `image` field; details: {e}"}, 404
   return { "result": model.predict(data).tolist()}

# start the development server
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')




