from flask import Flask, jsonify, request
import pickle
import os
from flask_cors import CORS, cross_origin
import numpy as np

# pip install -U flask-cors

# initialize our Flask application
app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = pickle.load(open('wine.pickle', 'rb'))


@app.route("/")
@cross_origin()
def helloWorld():
    return "Hello, cross-origin-world!"


@app.route("/wine", methods=["GET"])
def predWine():

    if request.method == 'GET':

        a = float(request.args.get('alcohol'))
        b = float(request.args.get('malic_acid'))
        c = float(request.args.get('ash'))
        d = float(request.args.get('alcalinity_of_ash'))
        e = float(request.args.get('magnesium'))
        f = float(request.args.get('total_phenols'))
        g = float(request.args.get('flavanoids'))
        h = float(request.args.get('nonflavanoid_phenols'))
        i = float(request.args.get('proanthocyanins'))
        j = float(request.args.get('color_intensity'))
        k = float(request.args.get('hue'))
        l = float(request.args.get('od315_of_diluted_wines'))
        m = float(request.args.get('proline'))

        final_features = ([[a, b, c, d, e, f, g, h, i, j, k, l, m]])

        prediction = model.predict(final_features)

    return jsonify(str("Class  " + str(prediction[0])))


#  main thread of execution to start the server
if __name__ == '__main__':
    app.run(debug=True)
