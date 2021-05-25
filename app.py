from flask import Flask, render_template, request

app = Flask(__name__)

import pickle
import numpy as np

model = pickle.load(open("model.pkl", 'rb'))

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def submit():
    if request.method == 'POST':
        cough = request.form['cough']
        fever = request.form['fever']
        sore = request.form['sore_throat']
        breath = request.form['breath']
        headache = request.form['headache']
        age = request.form['age']
        gender = request.form['gender']

        lst = [cough, fever, sore, breath, headache, age, gender]
        df = np.array(lst)
        df = df.reshape(1, 7)
        ans = model.predict(df)
        if ans == 1:
            return render_template("Yes.html")
        else:
            return render_template("No.html")
        # ans = np.array_str(ans)[1]

if __name__ == '__main__':
    app.run(debug=True)
