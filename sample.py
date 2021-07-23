import numpy as np
#import tensorflow as tf
import keras
from keras.models import load_model
from keras.preprocessing import image
import operator
import cv2
import sys, os

loaded_model = load_model("asl_model.h5")
# load weights into new model
#loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

# Category dictionary
categories = {0: 'A',
 1: 'B',
 2: 'C',
 3: 'D',
 4: 'E',
 5: 'F',
 6: 'G',
 7: 'H',
 8: 'I',
 9: 'J',
 10: 'K',
 11: 'L',
 12: 'M',
 13: 'N',
 14: 'O',
 15: 'P',
 16: 'Q',
 17: 'R',
 18: 'S',
 19: 'T',
 20: 'U',
 21: 'V',
 22: 'W',
 23: 'X',
 24: 'Y',
 25: 'Z',
 26: 'del',
 27: 'nothing',
 28: 'space'}


TEST_DIR = 'asl_alphabet_test/asl_alphabet_test/'
for img in os.listdir(TEST_DIR):
    temp = image.load_img(TEST_DIR+'/'+img,target_size=(200,200,3))
    temp = image.img_to_array(temp)
    temp = np.expand_dims(temp,axis=0)
    res = loaded_model.predict(temp)
    print(img.split('_')[0],categories[res.argmax()])



while True:
    _, frame = cap.read()
    # Simulating mirror image
    #frame = cv2.flip(frame, 1)

    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = 10
    y1 = 10
    x2 = int(0.5*frame.shape[1]) 
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]

    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (200, 200))
    
    #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #_, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)

    # Batch of 1


    '''temp = image.load_img(test_image)
    temp = image.img_to_array(temp)
    temp = np.expand_dims(temp,axis=0)
    res = loaded_model.predict(temp)'''
    test_image = roi

    #print(test_image.shape)
    #test_image = np.stack((test_image,)*3, axis=-1)
    #print(test_image.shape)
    #test_image = cv2.resize(test_image,(200,200))
    #print(test_image.shape)
    
    result = loaded_model.predict(test_image.reshape(1, 200, 200, 3))

    if result[0][0] == 1:
           print('A')
    elif result[0][1] == 1:
           print( 'B')
    elif result[0][2] == 1:
           print( 'C')
    elif result[0][3] == 1:
           print( 'D')
    elif result[0][4] == 1:
           print( 'E')
    elif result[0][5] == 1:
           print( 'F')
    elif result[0][6] == 1:
           print( 'G')
    elif result[0][7] == 1:
           print( 'H')
    elif result[0][8] == 1:
           print( 'I')
    elif result[0][9] == 1:
           print( 'J')
    elif result[0][10] == 1:
           print( 'K')
    elif result[0][11] == 1:
           print( 'L')
    elif result[0][12] == 1:
           print( 'M')
    elif result[0][13] == 1:
           print( 'N')
    elif result[0][14] == 1:
           print( 'O')
    elif result[0][15] == 1:
           print( 'P')
    elif result[0][16] == 1:
           print( 'Q')
    elif result[0][17] == 1:
           print( 'R')
    elif result[0][18] == 1:
           print( 'S')
    elif result[0][19] == 1:
           print( 'T')
    elif result[0][20] == 1:
           print( 'U')
    elif result[0][21] == 1:
           print( 'V')
    elif result[0][22] == 1:
           print( 'W')
    elif result[0][23] == 1:
           print( 'X')
    elif result[0][24] == 1:
           print( 'Y')
    elif result[0][25] == 1:
           print( 'Z')
    elif result[0][26] == 1:
           print( 'del')
    elif result[0][27] == 1:
           print( 'nothing')
    elif result[0][28] == 1:
           print( 'space')


    cv2.imshow("ROI",roi)
    cv2.imshow("Frame", frame)
    cv2.putText(frame, categories[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)

    interrupt = cv2.waitKey(50)
    if interrupt & 0xFF ==27: # esc key
        break



cap.release()
cv2.destroyAllWindows()
