import numpy as np 
import cv2
from PIL import ImageGrab
from keras.utils.data_utils import get_file
import dlib
from contextlib import contextmanager
from pathlib import Path
from wide_resnet import WideResNet

agegenderdetectmodel = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'
depth = 16
k = 8
margin = 0.4
weight_file = get_file("weights.28-3.73.hdf5", agegenderdetectmodel, cache_subdir="pretrained_models", file_hash=modhash, cache_dir="F:\screenface\pretrained_model")
model = WideResNet(64, depth = depth, k=k)()
model.load_weights(weight_file)

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
    img_h, img_w, _ = np.shape(frame)

    agendetected = detektoragen(frame, 1)
    faceagen = np.empty((len(agendetected), 64, 64, 3))
        
    if len(agendetected) > 0:
        for i, d in enumerate(agendetected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)
            faceagen[i, :, :, :] = cv2.resize(img_np[yw1:yw2 + 1, xw1:xw2 + 1, :], (64, 64))

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
