# Import the required libraries

from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()


for fn in uploaded.keys():

  # print("\n\nThe key is: "+ fn+"\n\n")
  # Predicting images
  # Full path for the image
  path = '/content/' + fn
  # Load the image into the img variable
  img = image.load_img(path, target_size =(180,180))
  x = image.img_to_array(img) # Convert the image into array
  x = np.expand_dims(x, axis=0) # Expand the images as if they were many images

  images = np.vstack([x]) # Important
  classes = model.predict(images, batch_size=10)  # Gives the predictions for each classes like: eg:  [classes[0]: ([ 3.729408   -0.32879975  0.60124546 -3.0891216   0.56141305])]
  score = tf.nn.softmax(classes[0]) #Put the predictions into a softmax layer to get a [score: (tf.tensor)]
  # print(score)
  # print(classes[0])


  # np.argmax(): Indicies of the max of the predictions
  # Finally to get the name of the predicted class we write the next line i.e the class_names[index(found above)]
  name = class_names[np.argmax(score)]
  accuracy = 100 * np.max(score) # Accuracy percentage

  plt.imshow(img)
  plt.axis("off")
  print("\n\nFinal prediction: ")
  print("The image is most likely "+ name +" with the accuracy of " + str(accuracy) )
