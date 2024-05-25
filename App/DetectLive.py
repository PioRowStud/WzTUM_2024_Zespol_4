import cv2
import torch
from tkinter import filedialog
from App.DetectInImage import predict

def live_video_capture(mlpModel,convModel,conv2Model,mode):
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ret, frame = cap.read()
        cv2.imshow('live camera', frame)
        if cv2.waitKey(30) == ord('q'):
            break
        if cv2.waitKey(30) == 32:
            predict(mlpModel,convModel,conv2Model,mode,image=frame)

    cap.release()
    cv2.destroyWindow("live camera")
