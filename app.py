from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    # header=None ensures first row is treated as data
    data = pd.read_csv(file, header=None)
    prediction = model.predict(data)
    result = ["The object is Rock." if p == 'R' else "The object is Mine." for p in prediction]
    return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
