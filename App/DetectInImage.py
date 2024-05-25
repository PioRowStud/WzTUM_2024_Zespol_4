import cv2
import torch
import tensorflow
from tkinter import filedialog


def predict(mlMmodel,convModel,conv2Model,mode,**kwargs):
    if "image" in kwargs:
        img=kwargs["image"]
    else:
        path=SelectImage()
        if path=="":
            return
        img = cv2.imread(path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(96, 96)
    )
    for (x, y, w, h) in face:
        scaled_face=cv2.resize(gray_image[y:y+h, x:x+w],(96,96),interpolation=cv2.INTER_LINEAR)
        if mode=="MLP":
            res=MlpPredict(scaled_face,mlMmodel)
        elif mode=="Conv1":
            res=Conv1Predict(scaled_face,convModel)
        elif mode=="Conv2":
            res = Conv2Predict(scaled_face,conv2Model)


        for idx in range(0,30,2):
            feat_x,feat_y = res[idx], res[idx+1]
            feat_x = (feat_x+1)*w/2
            feat_y = (feat_y+1)*h/2
            middle = (int(feat_x+x),int(feat_y+y))
            cv2.circle(img, middle, 5, color=(0, 0, 255),thickness=1, lineType=cv2.LINE_AA)

    ratio = img.shape[1] / img.shape[0]
    max_width = 800
    max_height = 600

    if ratio > 1:
        width = min(img.shape[1], max_width)
        height = int(width / ratio)
    else:
        height = min(img.shape[0], max_height)
        width = int(height * ratio)
    img=cv2.resize(img,(width,height),interpolation=cv2.INTER_LINEAR)
    cv2.imshow('features', img)
    cv2.waitKey()
    cv2.destroyWindow("features")
    return None

def MlpPredict(scaled_face,model):
    inp = torch.tensor(scaled_face.flatten(), dtype=torch.float64) / 255
    with torch.no_grad():
        res = model(inp)
    return res.numpy()
def Conv1Predict(scaled_face,model):
    a = scaled_face.reshape(-1, 96, 96, 1) / 255.0
    res = model.predict(a)
    return (res[0]/48)-1
def Conv2Predict(scaled_face,model):
    a = scaled_face.reshape(-1, 96, 96, 1) / 255.0
    res = model.predict(a)
    return (res[0] / 48) - 1
def SelectImage():
    return filedialog.askopenfilename()
