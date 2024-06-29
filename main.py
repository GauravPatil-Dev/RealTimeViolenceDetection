import cv2
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect
from flask_mail import Mail, Message
import os
import glob
import numpy as np
from extractor import Extractor
from keras.models import load_model
import sys
from subprocess import call
import shutil
import matplotlib
from keras import backend as K
import string 
import random 

matplotlib.use('Agg')

app = Flask(__name__)
app.config.update(dict(
    DEBUG=True,
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USE_SSL=False,
    MAIL_USERNAME='__recipient__email',
    MAIL_PASSWORD='__recipient__password',
))
mail = Mail(app)

K.clear_session()

inception = Extractor()
saved_model = "/home/gaurav/Downloads/eye_in_the_sky_latest/checkpoints/lstm-features-final.keras"
model = load_model(saved_model)

@app.route("/")
def home():    
    return render_template("home.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if request.form["video"] == "":
        print("No args! exiting")
        return redirect("/")

    seq_length = 40

    fname_ext = request.form["video"]
    file_path = "static/sample_videos/" + fname_ext
    fname = fname_ext.split('.')[0]
    
    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.")
        return redirect("/")

    call(["ffmpeg", "-i", file_path, "-r", str(seq_length),
          os.path.join('static/extracted_frames', fname + '-%04d.jpg')])

    frames = sorted(glob.glob(os.path.join(
        'static/extracted_frames', fname + '*jpg'))) 

    nframes = len(frames)-(len(frames) % seq_length)
    frames = frames[: nframes]
    
    if len(frames) == 0:
        print("No frames extracted.")
        return redirect("/")

    x = [i for i in range(0, nframes//seq_length)]
    y_violent = []
    y_non_violent = []

    # Extract features for all frames at once
    all_features = inception.extract_features(frames)

    for i in range(0, nframes, seq_length):
        sequence = all_features[i:i+seq_length]
        prediction = model.predict(np.expand_dims(sequence, axis=0))
        y_violent.append(prediction[0][1])
        y_non_violent.append(prediction[0][0])

    print(x)
    print(y_violent)
    print(y_non_violent)

    flag = 0
    for i in range(len(y_violent)):
        if(y_violent[i]>=0.85):
            start_frame_pos = i
            flag = 1
            break
    if flag:
        start_frame_pos = start_frame_pos * 40
        if(start_frame_pos == 0):
            start_frame_pos = 1
        end_frame_pos = start_frame_pos + 5
    
    avg_violence_score = sum(y_violent) / len(y_violent)
    print("avg_violence_score", avg_violence_score)

    plt.plot(x, y_violent, 'r', label='violence-score')
    plt.xlabel('time(s)')
    plt.ylabel('violence')
    plt.title('Violence in video')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig('static/plots/' + fname + '.png')
    plt.close()

    if(avg_violence_score>=0.85):
        msg = Message("Violence Detected", sender="hellomaneeshp@gmail.com", recipients=["__recipient__email"])
        msg.html = "<h3>Real Time Violence Detection System Alert</h3>"
        with app.open_resource("static/plots/" + fname +".png") as fp:
            msg.attach("violence_score_plot.png", "image/png", fp.read())
            for i in range(start_frame_pos+1,end_frame_pos+1):
                if(i < 10):
                    name = "000" + str(i) 
                if(i >=10 and i<100):
                    name = "00" + str(i) 
                if(i>=100):
                    name = "0" + str(i) 
                frame_name = fname + "-" + name        
                with app.open_resource("static/extracted_frames/" + frame_name +".jpg") as fp:
                    msg.attach("frame.jpg", "image/jpg", fp.read())
            #mail.send(msg)
    relative_file_path = os.path.join('sample_videos', fname_ext)     
    return render_template("analyze.html", file_path=relative_file_path, fname=fname, y_violent=y_violent)

@app.route("/live", methods=["POST"])
def live():
    seq_length = 40      
        
    chars = string.ascii_letters
    size = 5
    
    fname = 'live'.join(random.choice(chars) for x in range(size))

    print("The generated random file name : " + str(fname)) 

    video_path = f'static/live_videos/{fname}.mp4'
    call(
    [
        "ffmpeg",
        "-f", "v4l2",
        "-i", "/dev/video0",
        "-t", "5",
        video_path,
    ]
    )

    call(
    [
        "ffmpeg",
        "-i", video_path,
        "-r", str(seq_length),
        f"static/extracted_frames/{fname}-%04d.jpg",
    ]
    )

    frames = sorted(glob.glob(os.path.join(
        'static/extracted_frames', fname + '*jpg')))

    nframes = len(frames)-(len(frames) % seq_length)
    frames = frames[: nframes]

    x = [i for i in range(0, nframes//seq_length)]
    y_violent = []
    y_non_violent = []

    # Extract features for all frames at once
    all_features = inception.extract_features(frames)

    for i in range(0, nframes, seq_length):
        sequence = all_features[i:i+seq_length]
        prediction = model.predict(np.expand_dims(sequence, axis=0))
        y_violent.append(prediction[0][1])
        y_non_violent.append(prediction[0][0])

    print(x)
    print(y_violent)
    print(y_non_violent)

    plt.plot(x, y_violent, 'r', label='violence-score')
    plt.xlabel('time(s)')
    plt.ylabel('violence')
    plt.title('Violence in video')
    plt.ylim(0, 1)
    plt.legend()    
    plt.savefig('static/plots/' + fname + '.png')
    plt.close()

    return render_template("live.html", fname=fname,video_path=video_path, y_violent=y_violent)

if __name__ == "__main__":    
    app.run(debug=False, threaded=False)