#!flask/bin/python
################################################################################################################################
#------------------------------------------------------------------------------------------------------------------------------                                                                                                                             
# This file implements the REST layer. It uses flask micro framework for server implementation. Calls from front end reaches 
# here as json and being branched out to each projects. Basic level of validation is also being done in this file. #                                                                                                                                             
#-------------------------------------------------------------------------------------------------------------------------------                                                                                                                              
################################################################################################################################
from flask import Flask, jsonify, abort, request, make_response, url_for,redirect, render_template

from werkzeug.utils import secure_filename
import os
import sys
import random
import re
from tensorflow.python.platform import gfile
from six import iteritems
import shutil 
import numpy as np
import cv2
import tarfile

from datetime import datetime
from scipy import ndimage
from scipy.misc import imsave 
import tensorflow as tf

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
from tensorflow.python.platform import gfile

app = Flask(__name__, static_url_path = "")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
from data_util import DataLoader
from model import OCRnet
datadir="./Dataset2"
classes=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
ocrnet=OCRnet(len(classes),classes)
#==============================================================================================================================
#                                                                                                                              
#    This function is used to recognize uploaded images.                                                                       
#                                                                                                                              
#==============================================================================================================================
@app.route('/fileUpload', methods=['GET', 'POST'])
def upload_file():
    print("file upload")
    if request.method == 'POST' or request.method == 'GET':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        inputloc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image = cv2.imread(inputloc)#cv2.IMREAD_GRAYSCALE
        image=image/255
        ocrnet=OCRnet(len(classes),classes)
        ocrnet.load_model("./ocrnetmodel2.model")
        score,label = ocrnet.predict_image(image)
        print("score:", score)
        data = {
            'label':label,
            'score':score,   
        }
        return jsonify(data)
#==============================================================================================================================
#                                                                                                                              
#                                           Main function                                                                                                                                                
#                                                                                                                  
#==============================================================================================================================
@app.route("/")
def main():
    
    return render_template("index.html")   
if __name__ == '__main__':
    app.run(debug = True)