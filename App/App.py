import tkinter as tk
import os
import MLPModel.Model_2 as Model_2
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras import regularizers
from keras.losses import MeanSquaredError as mse
from tensorflow.keras import layers, models
import torch
from App.DetectInImage import predict
from App.DetectLive import live_video_capture

class App:

    def __init__(self):
        self.mode="MLP"
        self.root = tk.Tk()
        self.root.title('Key Facial features detection')
        mlp_model = self.loadMLPModel()
        conv_model = self.loadConv1Model()
        conv2_model = self.loadConv2Model()
        self.root.configure(background="white")
        tk.Label(
            self.root,text="Key Facial features\n detection",
            bg="white",
            fg="red",
            font=("Arial", 25)
        ).pack(pady=10)
        tk.Button(
            self.root,
            text="From image",
            width=25,
            height=5,
            bg="green",
            fg="yellow",
            command=lambda: predict(mlp_model, conv_model, conv2_model, self.mode)
        ).pack(pady=10)
        tk.Button(
            self.root,
            text="live camera",
            width=25,
            height=5,
            bg="green",
            fg="yellow",
            command=lambda: live_video_capture(mlp_model, conv_model,conv2_model, self.mode)
        ).pack(pady=10)
        self.modelLabel=tk.Label(
            self.root, text="current model\n MLP model",
            bg="white",
            fg="red",
            font=("Arial", 10)
        )
        self.modelLabel.pack(pady=10)
        tk.Button(
            self.root,
            text="change model",
            width=25,
            height=5,
            bg="blue",
            fg="black",
            command=lambda :self.changeModel()
        ).pack(pady=10)
    def changeModel(self):
        if self.mode=="MLP":
            self.mode="Conv1"
            self.modelLabel["text"] = "current model\n Conv1 model"
        elif self.mode == "Conv1":
            self.mode = "Conv2"
            self.modelLabel["text"] = "current model\n Conv2 model"
        else :
            self.mode = "MLP"
            self.modelLabel["text"] = "current model\n MLP model"
    def run(self):
        self.root.mainloop()


    def loadMLPModel(self):
        model = Model_2.MLP(9216, 30).double()
        cwd = os.getcwd()
        model.load_state_dict(torch.load(os.path.join(cwd, 'MLPModel', 'Model2Model.pth')))
        return model
    def loadConv1Model(self):
        cwd = os.getcwd()
        model = tf.keras.models.load_model(os.path.join(cwd,'Model_tf','model_edited.h5'))
        return model

    def loadConv2Model(self):
        cwd = os.getcwd()
        model = tf.keras.models.load_model(os.path.join(cwd, 'Model_tf', 'model2_edited.h5'))
        return model