from flask import Flask
from flask import render_template
from flask import request
from tensorflow import keras
import numpy as np
import json
from PIL import Image
import keras.utils as image

import os

app = Flask(__name__)
UPLOAD_FOLDER = "static"
dic = {0 : 'Parasitized', 1 : 'Uninfected'}

@app.route("/", methods = ["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            '''
            
            img = Image.open(image_file)


            
            img = img.resize((512, 512))
            img_array = np.array(img)
            
            print("SHAPE OF img Array 1",img_array.shape)
            img_array = img_array.astype('float32') / 255.0
            print("SHAPE OF img Array 2",img_array.shape)
            img_array = np.expand_dims(img_array, axis=0)
            print("SHAPE OF img Array 3 ",img_array.shape)
            img_array = np.reshape(img_array, (-1, 512))
            print("SHAPE OF img Array 4",img_array.shape)
            img_array = img_array.mean(axis=0, keepdims=True)
            print("SHAPE OF img Array 5 ",img_array.shape)
            '''

            model = keras.models.load_model('./model_3000.h5')
            print("Summary of the model", model.summary())
            
            i = image.load_img(image_location, target_size=(100,100))
            i = image.img_to_array(i)/255.0
            i = i.reshape(1, 100,100,3)
            prediction = model.predict(i)
            #predicted_labels = np.argmax(p, axis=1)
            #print("PREDICTED LABEL", predicted_labels)
            print("Selected image is predicted as: ", prediction)
            
           
            #train_datagen = keras.preprocessing.image.ImageDataGenerator()# no data augmentation for train set we did that in seperate code for class 1,3 and 4
            #prediction = model.predict(img_array)
            predicted_labels = np.argmax(prediction, axis=1)

            print(json.dumps({'prediction': (prediction.tolist())}))
            print("LAST PREDICTON is", prediction.tolist()[0])

            print(json.dumps({'prediction LABEL': predicted_labels.tolist()}))

            return render_template("index.html", prediction = dic[predicted_labels.tolist()[0]], 
                                                 pred_ratio = round(prediction[0][predicted_labels.tolist()[0]]*100, 2),
                                                 image_loc = image_file.filename)
        
    return render_template("index.html", prediction = 0, image_loc=None)


#if __name__ == "__main__":
 #   app.run(port=5007, debug=True)
