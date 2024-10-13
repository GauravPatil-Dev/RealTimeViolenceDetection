import cv2
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request, redirect, jsonify
from flask_socketio import SocketIO, emit
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
import threading
from collections import defaultdict, deque
import time
from queue import Queue
import json
from datetime import datetime

matplotlib.use('Agg')
import torch
from collections import deque
from datetime import datetime
from threading import Lock
from flask_cors import CORS
import logging
import warnings
# Locks for thread safety
violence_lock = Lock()
violation_lock = Lock()

# At the top of your file
violent_incidents = []
safety_violations = []

# Buffer to store frames for violence incidents
violence_frame_buffer = deque(maxlen=160)  # Stores frames for 4 seconds (160 frames at 40 fps)
violation_frame_buffer = deque(maxlen=20)  # Stores frames for 0.5 seconds (20 frames at 40 fps)

logging.getLogger("torch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch.cuda.amp.autocast.*")

# Load custom YOLOv5 model
yolo_model = torch.hub.load("/home/gaurav/Grainger_Dataset/yolo/unified_data/yolov5", "custom", path="/home/gaurav/Grainger_Dataset/yolo/unified_data/yolov5/runs/train/exp/weights/best.pt", source="local")

'''
app = Flask(__name__)
app.config.update(dict(
    DEBUG=True,
    SECRET_KEY='secret!',
))
socketio = SocketIO(app)
'''
app = Flask(__name__)
CORS(app, resources={r"/": {"origins": ""}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')



K.clear_session()

inception = Extractor()
saved_model = "/home/gaurav/RealTimeViolenceDetection/data/Checkpoints/lstm-features-final.keras"
model = load_model(saved_model)

# Global variables for frame buffer and processing
frame_buffer = deque(maxlen=200)  # 5 seconds at 40 fps
is_capturing = False
is_processing = False

# Add these global variables at the top of your file
frame_queue = Queue(maxsize=200)
analysis_thread = None

# Add these global variables
yolo_frame_queue = Queue(maxsize=10)  # Adjust size as needed
yolo_processed_frame_queue = Queue(maxsize=200)  # Adjust size as needed
'''
@app.route("/")
def home():
    return render_template("home.html")
'''

@app.route("/analyze", methods=["POST"])
def analyze():
    if request.form["video"] == "":
        print("No args! exiting")
        return redirect("/")

    seq_length = 40

    fname_ext = request.form["video"]
    file_path = "/home/gaurav/RealTimeViolenceDetection/static/sample_videos/" + fname_ext 
    fname = fname_ext.split('.')[0]
    print("File path: ", fname_ext)
    print("File path: ", file_path)
    print("File name: ", fname)
    
    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.") 
        return redirect("/")

    #call(["ffmpeg", "-i", file_path, "-r", str(seq_length),
    #      os.path.join('static/extracted_frames', fname + '-%04d.jpg')])
    call(["ffmpeg", "-i", file_path, "-r", str(seq_length), "-pix_fmt", "yuv420p", os.path.join('/home/gaurav/RealTimeViolenceDetection/static/extracted_frames', fname + '-%04d.jpg')])


    frame_paths = sorted(glob.glob(os.path.join(
        '/home/gaurav/RealTimeViolenceDetection/static/extracted_frames', fname + '*jpg')))


    nframes = len(frame_paths)-(len(frame_paths) % seq_length)
    #frames = frames[: nframes]
    frame_paths = frame_paths[: nframes]


    if len(frame_paths) == 0:
        print("No frames extracted.")
        return redirect("/")

    x = [i for i in range(0, nframes//seq_length)]
    y_violent = []
    y_non_violent = []
    frames = [cv2.imread(frame_path) for frame_path in frame_paths]

    # Extract features for all frames at once
    all_features = inception.extract_features(frames)

    class_colors = {
        0: (0, 255, 0),      # helmet - Green
        1: (255, 0, 0),      # not_helmet - Red
        2: (255, 0, 0),      # not_reflective - Red
        3: (0, 255, 255),    # reflective - Green
        4: (255, 0, 255),    # glove - Magenta
        5: (255, 0, 0),      # fall_detected - Red
        6: (128, 128, 0),    # walking - Olive
        7: (128, 0, 128),    # sitting - Purple
    }

    for i, frame in enumerate(frames):
        
        results = yolo_model(frame)
        detections = results.xyxy[0].cpu().numpy()
        print(detections)
        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            class_id = int(cls)
            color = class_colors.get(class_id, (0, 255, 0))  # Default to green if class ID not found
            label = f"{yolo_model.names[class_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save YOLO-annotated frame
        cv2.imwrite(os.path.join('/home/gaurav/RealTimeViolenceDetection/static/yolo_frames', f"{fname}-yolo-{i:04d}.jpg"), frame)


        if i % seq_length == 0 and i + seq_length <= len(frames):
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
    plt.savefig('/home/gaurav/RealTimeViolenceDetection/static/plots/' + fname + '.png')
    plt.close()

    if(avg_violence_score>=0.85):
        #msg = Message("Violence Detected", sender="hellomaneeshp@gmail.com", recipients=["__recipient__email"])
        #msg.html = "<h3>Real Time Violence Detection System Alert</h3>"
        with app.open_resource("/home/gaurav/RealTimeViolenceDetection/static/plots/" + fname + ".png") as fp:
            #msg.attach("violence_score_plot.png", "image/png", fp.read())
            for i in range(start_frame_pos+1,end_frame_pos+1):
                if(i < 10):
                    name = "000" + str(i)
                if(i >=10 and i<100):
                    name = "00" + str(i)
                if(i>=100):
                    name = "0" + str(i)
                frame_name = fname + "-" + name
                #with app.open_resource("/home/gaurav/RealTimeViolenceDetection/static/extracted_frames/" + frame_name + ".jpg") as fp:
                    #msg.attach("frame.jpg", "image/jpg", fp.read())
            #mail.send(msg)
    relative_file_path = os.path.join('sample_videos', fname_ext)
    return render_template("analyze.html", file_path=relative_file_path, fname=fname, y_violent=y_violent)

@app.route("/live", methods=["POST"])
def live():
    return render_template("live.html")

@socketio.on('start_stream')
def start_stream():
    global is_capturing, x, y_violent, y_non_violent
    chars = string.ascii_letters
    size = 5
    fname = 'live' + ''.join(random.choice(chars) for x in range(size))
    print("The generated random file name : " + str(fname))

    video_path = f'/home/gaurav/RealTimeViolenceDetection/static/live_videos/{fname}.mp4'
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 40.0, (640,480))

    is_capturing = True
    x = []
    y_violent = []
    y_non_violent = []
    threading.Thread(target=capture_frames, args=(out,)).start()
    threading.Thread(target=process_frames, args=(out,)).start()  # Pass 'out' to process_frames
    threading.Thread(target=yolo_processing).start()

    emit('stream_started', {'video_path': f'../{video_path}'})
safety_violation_buffer = deque(maxlen=200)  # 10 seconds of frames at 20 fps

def capture_frames(out):
    global frame_buffer, frame_queue, yolo_frame_queue, is_capturing
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 40)  # Set capture to 40 fps
    frame_count = 0
    while is_capturing:
        ret, frame = cap.read()
        if ret:
            safety_violation_buffer.append(frame.copy())

            frame_count += 1
            # Add original frame to buffers for LSTM-CNN analysis and YOLOv5 processing
            _, buffer = cv2.imencode('.jpg', frame)
            frame_encoded = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('frame', {'frame': frame_encoded})
            frame_buffer.append(frame.copy())
            if not frame_queue.full():
                frame_queue.put(frame.copy())
            if frame_count % 2 == 0:
                if not yolo_frame_queue.full():
                    yolo_frame_queue.put(frame.copy())
        else:
            break

    cap.release()
    print("Capture stopped")

def process_frames(out):
    global is_processing, analysis_thread
    is_processing = True
    analysis_thread = threading.Thread(target=continuous_analysis)
    analysis_thread.start()
    frames_written = 0
    last_frame_time = time.time()


    while is_processing:
        if not yolo_processed_frame_queue.empty():
            frame = yolo_processed_frame_queue.get()
            # Write the processed frame to the video file
            frame = cv2.resize(frame, (640, 480))
            out.write(frame)
            frames_written += 1
            
            # Log progress every second
            current_time = time.time()
            if current_time - last_frame_time >= 1:
                #print(f"Frames written: {frames_written}, Queue size: {yolo_processed_frame_queue.qsize()}")
                last_frame_time = current_time


            # Encode frame with bounding boxes and send it to client
            #_, buffer = cv2.imencode('.jpg', frame)
            #frame_encoded = base64.b64encode(buffer).decode('utf-8')
            #socketio.emit('frame', {'frame': frame_encoded})
        else:
            time.sleep(0.01)  # Short sleep to prevent busy-waiting

        # Ensure all remaining frames are written
    while not yolo_processed_frame_queue.empty():
        frame = yolo_processed_frame_queue.get()
        frame = cv2.resize(frame, (640, 480))
        out.write(frame)

    out.release()  # Release the VideoWriter when processing is done
    print("Video saving completed")

def continuous_analysis():
    global frame_queue, is_processing
    while is_processing:
        if frame_queue.qsize() >= 10:
            window = []
            for _ in range(10):
                window.append(frame_queue.get())
            analyze_window(window)
        else:
            time.sleep(0.1)  # Short sleep to prevent busy-waiting

def analyze_window(window):
    global x, y_violent, y_non_violent, violence_frame_buffer
    # Convert frames to format expected by the model
    frames = [cv2.resize(frame, (299, 299)) for frame in window]
    features = inception.extract_features(frames)

    # Reshape features for LSTM input
    sequence = np.expand_dims(features, axis=0)

    # Make prediction
    prediction = model.predict(sequence)
    print(f"Prediction: {prediction}")
    violence_score = prediction[0][1]
    non_violence_score = prediction[0][0]

    # Append scores to lists
    y_violent.append(violence_score)
    y_non_violent.append(non_violence_score)
    x.append(len(x))  # Append the current time step

    print(f"Violence score: {violence_score}")
    socketio.emit('violence_score', {'score': float(violence_score)})
    
    # Append current window frames to the buffer
    violence_frame_buffer.extend(window)


    if violence_score >= 0.25:
        alert_violence(violence_score)
        save_violence_frames()


    # Generate and emit the plot
    generate_and_emit_plot()

def save_violence_frames():
    global violence_frame_buffer,violent_incidents

    # Directory to save violent incident frames
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    incident_dir = os.path.join('/home/gaurav/RealTimeViolenceDetection/static/violence_frames', datetime.now().strftime('%Y%m%d%H%M%S'))
    os.makedirs(incident_dir, exist_ok=True)

    # Convert buffer to list to access frames by index
    frames_list = list(violence_frame_buffer)

    # Save frames from the buffer
    for idx, frame in enumerate(frames_list):
        cv2.imwrite(os.path.join(incident_dir, f"violent-{idx:04d}.jpg"), frame)

    print(f"Saved violent incident frames to {incident_dir}")

    # Add to incidents list
    with violence_lock:

        violent_incidents.append({'timestamp': timestamp, 'path': incident_dir})

    # Optionally, clear the buffer after saving
    violence_frame_buffer.clear()

def generate_and_emit_plot():
    plt.figure(figsize=(12, 8))
    plt.plot(x, y_violent, 'r', label='violence-score')
    plt.xlabel('time(s)')
    plt.ylabel('violence')
    plt.title('Violence in video')
    plt.ylim(0, 1)
    plt.legend()

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # Emit the plot data
    socketio.emit('plot_update', {'plot_data': plot_data})

@socketio.on('stop_stream')
def stop_stream():
    global is_capturing, is_processing, analysis_thread
    is_capturing = False
    is_processing = False
    if analysis_thread:
        analysis_thread.join()
    print("Stopping stream")
    emit('stream_stopped')

    # Clear the queues
    while not yolo_frame_queue.empty():
        yolo_frame_queue.get()
    while not yolo_processed_frame_queue.empty():
        yolo_processed_frame_queue.get()

    # The VideoWriter will be released in the process_frames function

@socketio.on('process_stream')
def process_stream():
    # This function is no longer needed as we're generating the plot in real-time
    pass

def alert_violence(violence_score):
    print(f"Violence detected! Score: {violence_score}")
    socketio.emit('violence_alert', {'score': float(violence_score)})
    # Here you can add code to send email alerts or trigger other notifications
    # For example:
    # send_email_alert(violence_score)
    # trigger_alarm_system()

# Add a new function for YOLOv5 processing
'''def yolo_processing():
    global yolo_frame_queue, yolo_processed_frame_queue, is_capturing

    while is_capturing:
        if not yolo_frame_queue.empty():
            frame = yolo_frame_queue.get()

            # YOLOv5 detection
            results = yolo_model(frame)  # Run YOLOv5 on the frame
            detections = results.xyxy[0].cpu().numpy()  # Get detections as NumPy array

            # Draw bounding boxes on the frame
            class_colors = {
                0: (0, 255, 0),      # helmet - Green
                1: (255, 0, 0),      # not_helmet - Red
                2: (255, 0, 0),      # not_reflective - Red
                3: (0, 255, 255),    # reflective - Green
                4: (255, 0, 255),    # glove - Magenta
                5: (255, 0, 0),      # fall_detected - Red
                6: (128, 128, 0),    # walking - Olive
                7: (128, 0, 128),    # sitting - Purple
            }

            for *xyxy, conf, cls in detections:
                x1, y1, x2, y2 = map(int, xyxy)
                color = class_colors.get(int(cls), (0, 255, 0))  # Default to green if class ID not found
                label = f"{yolo_model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Put processed frame in the queue
            if not yolo_processed_frame_queue.full():
                yolo_processed_frame_queue.put(frame)

        else:
            time.sleep(0.01)  # Short sleep to prevent busy-waiting
'''
def save_violation_frames():
    global violation_frame_buffer,safety_violations

    # Directory to save violation frames
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    violation_dir = os.path.join('/home/gaurav/RealTimeViolenceDetection/static/violation_frames', datetime.now().strftime('%Y%m%d%H%M%S'))
    os.makedirs(violation_dir, exist_ok=True)

    # Convert buffer to list
    frames_list = list(violation_frame_buffer)

    # Save frames from the buffer
    for idx, frame in enumerate(frames_list):
        cv2.imwrite(os.path.join(violation_dir, f"violation-{idx:04d}.jpg"), frame)

    #print(f"Saved safety violation frames to {violation_dir}")
    
    with violation_lock:
        safety_violations.append({'timestamp': timestamp, 'path': violation_dir})


    # Optionally, clear the buffer after saving
    violation_frame_buffer.clear()

def yolo_processing():
    global yolo_frame_queue, yolo_processed_frame_queue, is_capturing, yolo_model,violation_frame_buffer, safety_violation_buffer
    safety_compliance_over_time = []
    safety_violations_by_type = defaultdict(int)
    incident_timeline = []
    incident_frequency = defaultdict(lambda: defaultdict(int))

    # Initialize variables for data aggregation
    last_record_time = time.time()
    record_interval = 5  # seconds
    rolling_window = deque(maxlen=12)  # 1-minute rolling window at 5-second intervals
    last_detection_data = None
    interval_duration = 10  # 10 seconds for each buffer interval
    frames_per_interval = 10 * 20  # 10 seconds * 20 fps

    class_colors = {
        0: (0, 255, 0),      # helmet - Green
        1: (255, 0, 0),      # not_helmet - Red
        2: (255, 0, 0),      # not_reflective - Red
        3: (0, 255, 255),    # reflective - Green
        4: (255, 0, 255),    # glove - Magenta
        5: (255, 0, 0),      # fall_detected - Red
        6: (128, 128, 0),    # walking - Olive
        7: (128, 0, 128),    # sitting - Purple
    }

    while is_capturing:
        if len(safety_violation_buffer) >= frames_per_interval:
            print("Processing 10-second interval for safety violations...")
            frames_to_process = list(safety_violation_buffer)
            safety_violation_buffer.clear()  

        #if not yolo_frame_queue.empty():
        #    frame = yolo_frame_queue.get()
            #violation_frame_buffer.append(frame.copy())  # Add frame to buffer
            
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            violation_dir = os.path.join('/home/gaurav/RealTimeViolenceDetection/static/violation_frames', timestamp)
            os.makedirs(violation_dir, exist_ok=True)


            for idx, frame in enumerate(frames_to_process):
            # YOLOv5 detection
                results = yolo_model(frame)
                detections = results.xyxy[0].cpu().numpy()

                # Initialize data storage
                detection_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "camera_id": "webcam-0",
                    "location": "office",
                    "total_workers": 0,
                    "compliant_workers": 0,
                    "non_compliant_workers": 0,
                    "detections": []
                }

                # Tracking compliance flags
                workers_with_helmet = 0
                workers_with_reflective_vest = 0
                violations = defaultdict(int)
                incidents = defaultdict(int)
                
                # Process each detection
                violation_detected = False

                # Process each detection
                for *xyxy, conf, cls in detections:
                    x1, y1, x2, y2 = map(int, xyxy)
                    class_id = int(cls)
                    label = yolo_model.names[class_id]
                    confidence = float(conf)

                    # Append detection details
                    detection_data["detections"].append({
                        "object": label,
                        "confidence": confidence,
                        "bounding_box": [x1, y1, x2, y2]
                    })

                    # Draw bounding boxes on the frame
                    color = class_colors.get(class_id, (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


                    # Track compliance and violations
                    if label == "helmet":
                        workers_with_helmet += 1
                    elif label == "not_helmet":
                        violations["No Helmet"] += 1
                        violation_detected = True
                    elif label == "reflective":
                        workers_with_reflective_vest += 1
                    elif label == "not_reflective":
                        violations["No Reflective Vest"] += 1
                        violation_detected = True
                    elif label == "fall_detected":
                        incidents["Fall"] += 1
                        violation_detected = True
                        
                # Save violation frames if violation detected
                cv2.imwrite(os.path.join(violation_dir, f"frame-{idx:04d}.jpg"), frame)

                #if violation_detected:
                #    save_violation_frames()



            # Calculate totals and compliance
            detection_data["total_workers"] =max(workers_with_helmet + violations["No Helmet"],
                                                  workers_with_reflective_vest + violations["No Reflective Vest"])
            detection_data["compliant_workers"] = min(workers_with_helmet, workers_with_reflective_vest)
            detection_data["non_compliant_workers"] = detection_data["total_workers"] - detection_data["compliant_workers"]

            # Record data at intervals or if there's a significant change
            current_time = time.time()
            if (current_time - last_record_time >= record_interval) or \
               (last_detection_data and abs(detection_data['compliant_workers'] - last_detection_data['compliant_workers']) > 1):
                
                rolling_window.append(detection_data)
                last_record_time = current_time
                last_detection_data = detection_data

                # Calculate rolling averages
                avg_total = sum(d['total_workers'] for d in rolling_window) / len(rolling_window)
                avg_compliant = sum(d['compliant_workers'] for d in rolling_window) / len(rolling_window)
                # Update safety violations by type
                for violation_type, count in violations.items():
                    safety_violations_by_type[violation_type] += count

                # Update incident timeline and frequency
                for incident_type, count in incidents.items():
                    if count > 0:
                        incident_timeline.append({
                            "timestamp": current_time.isoformat(),
                            "incident_type": incident_type,
                            "count": count
                        })
                        incident_frequency[current_time.strftime("%A")][current_time.hour] += count


                aggregate_data = {
                    "timestamp": detection_data["timestamp"],
                    "camera_id": detection_data["camera_id"],
                    "location": detection_data["location"],
                    "avg_total_workers": avg_total,
                    "avg_compliant_workers": avg_compliant,
                    "avg_compliance_rate": avg_compliant / avg_total if avg_total > 0 else 1.0,
                    "current_total_workers": detection_data["total_workers"],
                    "current_compliant_workers": detection_data["compliant_workers"],
                    "current_compliance_rate": detection_data["compliant_workers"] / detection_data["total_workers"] if detection_data["total_workers"] > 0 else 1.0,
                    "violations": dict(violations),
                    "incidents": dict(incidents)
                }

                # Store aggregate data
                with open("aggregate_detection_data.json", "a") as f:
                    json.dump(aggregate_data, f)
                    f.write("\n")

                # Emit aggregate data through socketio
                socketio.emit('aggregate_data', aggregate_data)

            # Put processed frame in the queue for display or further processing
            if not yolo_processed_frame_queue.full():
                yolo_processed_frame_queue.put(frame)
                violation_frame_buffer.append(frame.copy())  # Add frame to buffer

                #print("Frame added to queue and size is", yolo_processed_frame_queue.qsize())

        else:
            time.sleep(0.01)  # Short sleep to prevent busy-waiting

    print("YOLO processing stopped")
if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)