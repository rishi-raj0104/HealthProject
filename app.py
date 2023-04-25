from flask import Flask, flash, request, redirect, url_for, render_template,jsonify
import urllib.request
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS,cross_origin
import cv2
import pickle
import imutils
import sklearn
from tensorflow.keras.models import load_model
# from pushbullet import PushBullet
import joblib
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
import xgboost as xgb


braintumor_model = load_model('D:/finalyear/Health/models/braintumor.h5')
pneumonia_model = load_model('D:/finalyear/Health/models/pneumonia_model.h5')
breastcancer_model = joblib.load('D:/finalyear/Health/models/cancer_model.pkl')

#static/uploads
UPLOAD_FOLDER = 'D:/finalyear/Health/static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

############################################# BRAIN TUMOR FUNCTIONS ################################################


def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(img,dsize=img_size,interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)


def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
                      extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)

##################################################################################################


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('homepage.html')


@app.route('/breastcancer')
def breast_cancer():
    return render_template('breastcancer.html')


@app.route('/braintumor')
def brain_tumor():
    return render_template('braintumor.html')

@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')

@app.route('/parkinson')
def parkinson():
    return render_template('parkinson.html')

###################################################################

@app.route('/resultbt', methods=['POST'])
def resultbt():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')#static/uploads/
            img = cv2.imread('D:/finalyear/Health/static/uploads/'+filename)
            img = crop_imgs([img])
            img = img.reshape(img.shape[1:])
            img = preprocess_imgs([img], (224, 224))
            pred = braintumor_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Brain Tumor test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
            return render_template('resultbt.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)


@app.route('/resultbc', methods=['POST'])
def resultbc():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        cpm = request.form['concave_points_mean']
        am = request.form['area_mean']
        rm = request.form['radius_mean']
        pm = request.form['perimeter_mean']
        cm = request.form['concavity_mean']
        pred = breastcancer_model.predict(
            np.array([cpm, am, rm, pm, cm]).reshape(1, -1))
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Breast Cancer test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resultbc.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

@app.route('/resultp', methods=['POST'])
def resultp():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')#static/uploads/
            img = cv2.imread('D:/finalyear/Health/static/uploads/'+filename)
            img = cv2.resize(img, (150, 150))
            img = img.reshape(1, 150, 150, 3)
            img = img/255.0
            pred = pneumonia_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour COVID-19 test results are ready.\nRESULT: {}'.format(firstname,['POSITIVE','NEGATIVE'][pred]))
            return render_template('resultp.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)

@app.route('/results', methods=['POST', 'GET']) 
def predict():
    if request.method == 'POST':
        try:
            mdvp_fo = float(request.form['mdvp_fo'])
            mdvp_fhi = float(request.form['mdvp_fhi'])
            mdvp_flo = float(request.form['mdvp_flo'])
            mdvp_jitper = float(request.form['mdvp_jitper'])
            mdvp_jitabs = float(request.form['mdvp_jitabs'])
            mdvp_rap = float(request.form['mdvp_rap'])
            mdvp_ppq = float(request.form['mdvp_ppq'])
            jitter_ddp = float(request.form['jitter_ddp'])
            mdvp_shim = float(request.form['mdvp_shim'])
            mdvp_shim_db = float(request.form['mdvp_shim_db'])
            shimm_apq3 = float(request.form['shimm_apq3'])
            shimm_apq5 = float(request.form['shimm_apq5'])
            mdvp_apq = float(request.form['mdvp_apq'])
            shimm_dda = float(request.form['shimm_dda'])
            nhr = float(request.form['nhr'])
            hnr = float(request.form['hnr'])
            rpde = float(request.form['rpde'])
            dfa = float(request.form['dfa'])
            spread1 = float(request.form['spread1'])
            spread2 = float(request.form['spread2'])
            d2 = float(request.form['d2'])
            ppe = float(request.form['ppe'])

            # Load the trained model and scaler
            model = joblib.load('D:/finalyear/Health/Parkinson/parkinsons_model.joblib')
            scaler = joblib.load('D:/finalyear/Health/Parkinson/scaler.joblib')

            # Transform the input data using the loaded scaler
            input_data = np.array([mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitper, mdvp_jitabs, mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shim, mdvp_shim_db, shimm_apq3, shimm_apq5, mdvp_apq, shimm_dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe])
            input_data = input_data.reshape(1, -1)
            input_data = scaler.transform(input_data)

            # Make predictions using the loaded model
            prediction = model.predict(input_data)

            if prediction == 1:
                pred = "You have Parkinson's Disease. Please consult a specialist."
            else:
                pred = "You are a healthy person."
                
            return render_template('results.html', prediction=pred)

        except Exception as e:
            print('The Exception message is: ', e)
            return 'Something went wrong.'
    else:
        return render_template('parkinson.html')

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)
