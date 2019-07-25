from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
import pandas as pd
from scipy import misc
import os
from conf.config import config

app = Flask(__name__)
api = Api(app)


img_dir = config.img_path
model_path = config.model_path

def load_model(dir):
    with open(dir,'rb') as f:
        return pickle.load(f)

def load_image_as_input(dir):
    maxint = 0
    for filename in os.listdir(img_dir):
        if filename.endswith(".jpg"):
            number = int(filename[:-4])
            if number > maxint:
                maxint = number
    image = misc.imread(img_dir+str(maxint)+'.jpg')

    return np.reshape(image,(1,32,32,3))

# load model and input
model = load_model(model_path)
X = load_image_as_input(img_dir)

# make prediction
answer = model.predict(X)


class HelloWorld(Resource):
    def get(self):
        y = np.reshape(answer,(10))
        y = np.argmax(y)
        yy=pd.Series(y).to_json(orient='values')

        return yy

api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(port=5001,debug=True)