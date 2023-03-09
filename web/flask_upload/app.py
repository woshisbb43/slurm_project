# 参考 https://medium.com/featurepreneur/uploading-files-using-flask-ec9fb4c7d438
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os 
import train
import monitor
import cmdexe

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/"

model, class_names = train.train_model()

@app.route('/')
def upload_file():
    squeue_info = cmdexe.getsqueue()
    metric_data = monitor.get_metrics()
    return render_template('index.html', metric_data=metric_data, squeue_info=squeue_info)

@app.route('/display', methods = ['GET', 'POST'])
def display_file():
    if request.method == 'POST':
        f = request.files['file']
        if not f:
            return 'Please select image to upload'
        filename = secure_filename(f.filename)
        f.save(app.config['UPLOAD_FOLDER'] + filename)
        file = open(app.config['UPLOAD_FOLDER'] + filename)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #########
        train_result = train.predict_img(model, class_names, full_filename)
        #########
    return render_template('content.html', user_image = full_filename, train_result=train_result) 

@app.route('/load/<path:loadtype>')
def load(loadtype):
    squeue_info = cmdexe.execute(loadtype)
    metric_data = monitor.get_metrics()
    return render_template('jobinfo.html',squeue_info=squeue_info, metric_data=metric_data)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug = True)