import cv2
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
import numpy as np
from multiprocessing import Pool

def process_frame(frame):
    # Resize the frame to (299, 299)
    img = cv2.resize(frame, (299, 299))
    # Convert BGR to RGB (OpenCV uses BGR by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

class Extractor():
    def __init__(self):
        self.weights = None 
        base_model = InceptionV3(
            weights='imagenet',
            include_top=True
        )

        self.model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('avg_pool').output
        )

    def extract_features(self, frames, batch_size=32):
        with Pool() as p:
            processed_frames = p.map(process_frame, frames)
        features = []
        for i in range(0, len(processed_frames), batch_size):
            batch = np.vstack(processed_frames[i:i+batch_size])
            batch_features = self.model.predict(batch)
            features.extend(batch_features)
        return np.array(features)

    def extract(self, frame):
        return self.extract_features([frame])[0]
