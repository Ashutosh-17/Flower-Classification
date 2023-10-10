import os
import numpy as np
import keras.utils as image
from flask import Flask, render_template, request, session
import pickle
from werkzeug.utils import secure_filename

# WSGI Application
# Defining upload folder path
UPLOAD_FOLDER = "static/upload"

# define allowed files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'This is your secret key to utilize session in Flask'

model = pickle.load(open('models/model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template("index.html")


@app.route("/", methods=['POST', 'GET'])
def uploadFile():
    if request.method == 'POST':
        # Upload file flask
        uploaded_img = request.files['img']

        # extracting uploaded data file name
        img_filename = secure_filename(uploaded_img.filename)

        # upload file to database (defined uploaded folder in static path)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))

        # storing uploaded file path in flask session
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)

        # Retrieving uploaded file path from session
        img_file_path = session.get('uploaded_img_file_path', None)

        img = image.load_img(img_file_path, target_size=(180, 180))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        result = model.predict(x)

        classes = os.listdir('dataset/training_set')
        index = int(np.argmax(result, axis=-1))

        return render_template(classes[index] + '.html')


if __name__ == "__main__":
    app.run(debug=True)
