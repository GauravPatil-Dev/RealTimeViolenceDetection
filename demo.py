# from data import DataSet
import os
import glob
import numpy as np
from extractor import Extractor
from keras.models import load_model
import sys
from subprocess import call
import shutil
import matplotlib.pyplot as plt

if len(sys.argv) == 1:
    print("No args! exiting")
    exit()

# shape that lstm expects is (40, 2048)
seq_length = 40

# load trained lstm custom model
inception = Extractor()  # load pre-trained inception model
saved_model = 'data/checkpoints/lstm-features.008-0.105.hdf5'
model = load_model(saved_model)


fname_ext = os.path.basename(sys.argv[1])
fname = fname_ext.split('.')[0]
# print(fname)
# print(sys.argv[1])
call(["ffmpeg", "-i", sys.argv[1], "-r", str(seq_length),
      os.path.join('data/extracted_frames', fname + '-%04d.jpg')])

# data = DataSet(seq_length=40, class_limit=2)
frames = sorted(glob.glob(os.path.join(
    'data/extracted_frames', fname + '*jpg')))


# make sure number of frames is a multiple of seq_length
nframes = len(frames)-(len(frames) % seq_length)

# remove extra frames
frames = frames[: nframes]

x = [i for i in range(0, nframes//seq_length)]
# x = np.linspace(0, nframes//40, nframes//40)
y_violent = []
y_non_violent = []
for i in range(0, nframes, seq_length):
    sequence = []
    for frame in frames[i: i+seq_length]:
        features = inception.extract(frame)
        sequence.append(features)
    prediction = model.predict(np.expand_dims(sequence, axis=0))
    # prediction[0][0] is non violent score
    # prediction[0][1] is violent score
    # print(prediction)
    y_violent.append(prediction[0][1])
    y_non_violent.append(prediction[0][0])
    # np.save('data/saved_sequence/' + fname, sequence)

print(x)
print(y_violent)
print(y_non_violent)

# plt.step(x, y_violent, label='violence score')
# plt.step(x, y_non_violent, label='non-violent score')
plt.plot(x, y_violent, 'r', label='violence-score')
plt.plot(x, y_non_violent, 'b', label='non-violence-score')

plt.xlabel('time(s)')
plt.ylabel('violence')
plt.title('Violence in video')
plt.ylim(0, 1)
plt.legend()
plt.savefig('data/plots/testplot.png')
# plt.show()
plt.close()

# clean up by deleting frames captured
shutil.rmtree('data/extracted_frames')
os.makedirs('data/extracted_frames')

# print(prediction)

# fig1, ax1 = plt.subplots()
# labels = ['Non violent score', 'Violent score']
# explode = [0, 0.1]
# ax1.pie(prediction.tolist()[0], explode=explode, labels=labels, autopct='%1.1f%%',
#         shadow=True, startangle=90)
# ax1.axis('equal')
# plt.show()
