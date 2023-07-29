from flask import Flask,request,render_template
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import tensorflow as tf
from keras.models import load_model
import io


model = load_model("u-net.h5", compile=False)
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

RESULT_FOLDER = 'static/results/'
app.config['RESULT_FOLDER'] = RESULT_FOLDER

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/home',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('image')
        if file is None:
            return "No file uploaded"
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        # Load image and convert to RGB
        img = Image.open(file)
        img = img.convert('RGB')

        # Resize image to expected shape
        img = img.resize((256, 256))

        # Convert image to numpy array and normalize
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0

        # Predict using the model
        predi = model.predict(np.expand_dims(img_array, axis=0))
        
        # Convert the predicted image to grayscale
        predi_gray = Image.fromarray(np.uint8(predi[0, :, :, 0] * 255))
        predi_gray = ImageOps.grayscale(predi_gray)
        
        # Save predicted image to a PNG file
        predi_gray.save(os.path.join(app.config['RESULT_FOLDER'], 'result.png'))
        # Pass path of the saved image to the template
        return render_template('result.html',input_image='static/uploads/'+ file.filename, result_image='static/results/result.png')

    else:
        return render_template("index.html")
    

if __name__ == '__main__':
    app.run(debug=True)
