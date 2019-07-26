from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
import pandas as pd
from scipy import misc
import os
from conf.config import config
import cv2

app = Flask(__name__)
api = Api(app)

img_dir = config.img_path
model_path = config.model_path

classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_model(dir):
    with open(dir,'rb') as f:
        return pickle.load(f)

def processing(data):
    height = 32
    width = 32
    data = data[:,:,0:3]
    dim = (width, height)
    res = cv2.resize(data, dim, interpolation = cv2.INTER_AREA)
    return(res)

def load_image_as_input(dir):
    
    image = misc.imread(img_dir+'target.jpg')
    image_process = processing(image)

    return np.reshape(image_process,(1,32,32,3))

# load model and input

model = load_model(model_path)
k = misc.imread('dev/test/test.jpg')
k = np.reshape(k,(1,32,32,3))
y = model.predict(k)

# make prediction


class HelloWorld(Resource):
    def get(self):
        X = load_image_as_input(img_dir)
        answer = model.predict(X)
        y = np.reshape(answer,(10))
        y = np.argmax(y)
        proj_y = classes[y]
        yy=pd.Series(proj_y).to_json(orient='values')

        return yy

api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(port=5001,debug=True)