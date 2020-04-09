import numpy as np 
import cv2
import os
import dlib
from PIL import ImageGrab
from contextlib import contextmanager
from pathlib import Path
from statistics import mode
from wide_resnet import WideResNet
from keras.models import load_model
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

depth = 16
k = 8
margin = 0.4
weight_file = "F:\screenface\pretrained_model\weights.28-3.73.hdf5"
emotion_file = "F:\screenface\pretrained_model\emotion_model.hdf5"
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = WideResNet(64, depth = depth, k=k)()
model.load_weights(weight_file)
emotion_labels = get_labels('fer2013')
emotion_class = load_model(emotion_file)
emotion_target_size = emotion_class.input_shape[1:3]
emotion_window = []
frame_window = 10
emotion_offsets = (20, 40)

detektoragen = dlib.get_frontal_face_detector()

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (200, 0, 200), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


while True:
    img = ImageGrab.grab()
    img_np = np.array(img)
    k = cv2.waitKey(1)
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    grayframe = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    img_h, img_w, _ = np.shape(frame)

    agendetected = detektoragen(frame, 1)

    faceagen = np.empty((len(agendetected), 64, 64, 3))

    faces = face_cascade.detectMultiScale(grayframe, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        
    if len(agendetected) > 0:
        for i, d in enumerate(agendetected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)
            faceagen[i, :, :, :] = cv2.resize(img_np[yw1:yw2 + 1, xw1:xw2 + 1, :], (64, 64))
        
        for d in faces:
            ex1, ex2, ey1, ey2 = apply_offsets(d, emotion_offsets)
            gray_face = grayframe[ey1:ey2, ex1:ex2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_preds = emotion_class.predict(gray_face)
            emotion_prob = np.max(emotion_preds)
            emotion_label_arg = np.argmax(emotion_preds)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)
            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            if emotion_text == 'angry':
                color = emotion_prob*np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_prob*np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_prob*np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_prob*np.asarray((0, 255, 255))
            else:
                color = emotion_prob*np.asarray((0, 255, 0))
            color = color.astype(int)
            color = color.tolist()
            draw_text(d, frame, emotion_mode, color, 0, 80, 1, 1)

        results = model.predict(faceagen)

        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()

        for i, d in enumerate(agendetected):
            label = "{}, {}".format(int(predicted_ages[i]), "M" if predicted_genders[i][0] < 0.5 else "F")
            
            draw_label(frame, (d.left(), d.top()), label)

    cv2.imshow("Screen", frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()