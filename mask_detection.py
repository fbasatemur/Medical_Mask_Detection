# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 00:02:02 2020

@author: fbasatemur
""" 

import cv2
import datetime
import imutils
import numpy as np

from keras.models import load_model
import cv2
import numpy as np

protopath = "./model/deploy.prototxt"
modelpath = "./model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
model = load_model("./model/model-017.model")

labels_dict = {0:"masked", 1:"unmasked"}
color_dict = {0:(0,255,0), 1:(0,0,255)}


def calculateFps(start_time, total_frames):
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - start_time

    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    return fps


def main():
    camera = cv2.VideoCapture(0)

    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0

    while True:
        ret, frame = camera.read()
        frame = imutils.resize(frame, width=600)
        total_frames = total_frames + 1

        (H, W) = frame.shape[:2]

        face_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

        detector.setInput(face_blob)
        face_detections = detector.forward()

        for i in np.arange(0, face_detections.shape[2]):
            confidence = face_detections[0, 0, i, 2]
            if confidence > 0.5:

                face_box = face_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = face_box.astype("int")
                
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                face_image = gray_image[startY:endY, startX: endX]
                resized = cv2.resize(face_image, (100, 100))
                  
                normalized = resized / 125
                reshaped = np.reshape(normalized, (1,100,100,1))
                result = model.predict(reshaped)
                # print(result)
                # print(np.argmax(result, axis= 1)[0])
                label = np.argmax(result, axis= 1)[0]
                  
                cv2.rectangle(frame, (startX, startY), (endX, endY), color_dict[label], 2)
                cv2.rectangle(frame, (startX, startY - 40), (endX, startY), color_dict[label], -1)
                cv2.putText(frame, labels_dict[label], (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                acc=round(np.max(result, axis=1)[0] * 100, 2)
                cv2.putText(frame, str(acc), (startX+140, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255), 2) 

        
        fps = calculateFps(fps_start_time, total_frames)
        fps_text = "FPS: {:.2f}".format(fps)
        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        cv2.imshow("Mask detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    camera.release()


if __name__ == '__main__':
    main()