import flask
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from fastai.vision import *
import os


# Use pickle to load in the pre-trained model.
learner=load_learner('/Users/soniarode/Desktop/ClassifierApp/model')

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('main.html')

@app.route('/prediction/<filename>')
def prediction(filename):
    #Step 1

    img = open_image('uploads/'+filename)


    pred_class,pred_idx,outputs = learner.predict(img)
    print(pred_class.obj)
    return render_template('predict.html', prediction=pred_class.obj)



if __name__ == '__main__':
    app.run()
