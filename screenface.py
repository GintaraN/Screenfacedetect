import numpy as np 
import cv2
from PIL import ImageGrab

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img_counter = 0

while True:
    img = ImageGrab.grab()
    img_np = np.array(img)
    k = cv2.waitKey(1)

    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Screen", frame)

    if k%256 == 27: 
        break
    elif k%256 == 32:
        img_name = "screenface_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} !".format(img_name))
        img_counter += 1

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
