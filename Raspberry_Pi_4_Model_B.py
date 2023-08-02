import RPi.GPIO as GPIO
import pyttsx3
# Set up the buttons
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# .....................................................................................................................
def text_to_speech(text):
  engine = pyttsx3.init()
  engine.say(text)
  engine.runAndWait()

# .......................................................................................................................................
import cv2
import pytesseract
import pyttsx3

# Set the path to the Tesseract OCR engine
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize the text-to-speech engine
engine = pyttsx3.init()


def read_text_from_camera():
  # Initialize the camera
  cap = cv2.VideoCapture(0)

  while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to the grayscale image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Apply OCR using pytesseract
    text = pytesseract.image_to_string(thresh)

    # If text is found, read it out using the text-to-speech engine
    if text:
      engine.say(text)
      engine.runAndWait()

    # Display the camera feed
    cv2.imshow('Camera Feed', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Release the camera and close the window
  cap.release()
  cv2.destroyAllWindows()


# .............................................................................................................................................

import cv2
import os
from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import tensorflow as tf
import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout,MaxPool2D,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

img_width, img_height = 150,150
batch_size = 32
epochs = 10

path ="D:/xxxxxx/indian_currency_new"
train_data_dir = "D:/xxxxxx/indian_currency_new/training"
test_data_dir = "D:/xxxxxxx/indian_currency_new/validation"

from glob import glob
glob("D:/xxxxxx/indian_currency_new/training/*/")

labels = ['10','100','20','200','2000','50','500']
trainGen = ImageDataGenerator(rescale=1./225, shear_range=0.2,horizontal_flip=True,zoom_range=0.2)
testGen = ImageDataGenerator(rescale=1./255)

train = trainGen.flow_from_directory(train_data_dir,target_size=(img_height,img_width),classes=labels,class_mode='categorical',batch_size=batch_size,shuffle=True)
test = testGen.flow_from_directory(test_data_dir,target_size=(img_height,img_width),classes=labels,class_mode= 'categorical',batch_size=batch_size)

model = Sequential()
model.add(Conv2D(128, (3,3),input_shape=(img_height,img_width,3),padding="same",activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64,3,3,padding="same",activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(32,3,3,padding="same",activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Dense(32,activation='relu'))

model.add(Flatten())
model.add(Dense(32,activation='relu'))

model.add(Dense(7,activation='softmax'))

model.summary()


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

epochs =15


model.fit(train,epochs=epochs,validation_data=test)
from keras.preprocessing import image
import keras.utils as image
from tensorflow.keras.utils import img_to_array


def predict_note(path):
  img_pred = image.load_img(path, target_size=(img_height, img_width))
  img_pred = image.img_to_array(img_pred)
  img = np.expand_dims(img_pred, axis=0)
  # result = model.predict_classes(img)
  predict_x = model.predict(img)
  result = np.argmax(predict_x, axis=1)

  #        prob= model.predict_proba(img)
  predict_prob = model.predict(img)
  prob = np.argmax(predict_prob, axis=1)
  if result[0] == 0:
    prediction = "10"
  elif result[0] == 1:
    prediction = "100"
  elif result[0] == 2:
    prediction = "20"
  elif result[0] == 3:
    prediction = "200"
  elif result[0] == 4:
    prediction = "2000"
  elif result[0] == 5:
    prediction = "50"
  else:
    prediction = "500"

  print('Prediction: ', prediction)
  text_to_speech(prediction)

# ..........................................................................................................................
# Define the functions that will be executed when the buttons are pressed
def button1_pressed():
  print("Button 1 pressed")
  read_text_from_camera()


def button1_released():
  print("Button 1 released")


def button2_pressed():
  print("Button 2 pressed")
  def take_photo(file_path):
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
      return
    ret, frame = cap.read()
    if not ret:
      return
    cv2.imwrite(file_path, frame)
    cap.release()
    print("Photo saved as", file_path)
  photo_file = "photo.jpg"
  take_photo(photo_file)
  predict_note("photo.jpg")



def button2_released():
  print("Button 2 released")
GPIO.add_event_detect(17, GPIO.FALLING, callback=button1_pressed)
GPIO.add_event_detect(27, GPIO.FALLING, callback=button2_pressed)

# Start the main loop
while True:
  GPIO.wait_for_edge(17, GPIO.FALLING)
  button1_pressed()
  while GPIO.input(17) == GPIO.LOW:
    pass
  button1_released()

  GPIO.wait_for_edge(27, GPIO.FALLING)
  button2_pressed()
  while GPIO.input(27) == GPIO.LOW:
    pass
  button2_released()