# Python program to transform an image using 
# threshold. 
import numpy as np 
import cv2 
from matplotlib import pyplot as plt 

import keras
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import os
import cv2
import imutils
import os
import time
from glob import glob
from twilio.rest import Client
clas1 = [item[13:-1] for item in sorted(glob("./validation/*/"))]


from keras.preprocessing import image                  
from tqdm import tqdm

# Note: modified these two functions, so that we can later also read the inception tensors which 
# have a different format 
def path_to_tensor(img_path, width=224, height=224):
    # loads RGB image as PIL.Image.Image type
    print(img_path)
    img = image.load_img(img_path, target_size=(width, height))
    # convert PIL.Image.Image type to 3D tensor with shape (width, heigth, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, width, height, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, width=224, height=224):
    list_of_tensors = [path_to_tensor(img_paths, width, height)]
    return np.vstack(list_of_tensors)



# callback to show the total time taken during training and for each epoch
class EpochTimer(keras.callbacks.Callback):
    train_start = 0
    train_end = 0
    epoch_start = 0
    epoch_end = 0
    
    def get_time(self):
        return timeit.default_timer()

    def on_train_begin(self, logs={}):
        self.train_start = self.get_time()
 
    def on_train_end(self, logs={}):
        self.train_end = self.get_time()
        print('Training took {} seconds'.format(self.train_end - self.train_start))
 
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start = self.get_time()
 
    def on_epoch_end(self, epoch, logs={}):
        self.epoch_end = self.get_time()
        print('Epoch {} took {} seconds'.format(epoch, self.epoch_end - self.epoch_start))

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

#from tkinter import filedialog
#filename = filedialog.askopenfilename(title='open')

#main_img = cv2.imread(filename)


from keras.models import load_model
model2 = load_model('trained_modelDNN1.h5')

from tkinter import filedialog
filename = filedialog.askopenfilename(title='open')

main_img = cv2.imread(filename)


# Image operation using thresholding 
image1= cv2.imread(filename) 

cv2.imshow('Original Image',image1)

img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", img)
# applying Otsu thresholding
# as an extra flag in binary 
# thresholding     
ret, mask = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + 
                                            cv2.THRESH_OTSU)     
# remove noise
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((8,8),dtype=np.uint8))
# apply mask to original image
result = cv2.bitwise_and(image1,image1,mask=mask)

#show image
cv2.imshow("Segmented", result)
cv2.imshow("Mask", mask)



#cv2.imshow('Given Image',main_img)
test_tensors = paths_to_tensor(filename)/255
pred=model2.predict(test_tensors)
pred=np.argmax(pred);
print('given Image Predicted as  = '+clas1[pred])

account_sid = 'from twilio'
auth_token = 'from twilio'
  
client = Client(account_sid, auth_token)
  
''' Change the value of 'from' with the number 
received from Twilio and the value of 'to'
with the number in which you want to send message.'''
message = client.messages.create(
                              from_='phone number from twilio',
                              body ='given Image Predicted as  = '+clas1[pred],
                              to ='your number'
                          )
