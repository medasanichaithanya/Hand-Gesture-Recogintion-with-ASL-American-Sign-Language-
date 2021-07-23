import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import operator
import cv2
import sys, os

# Loading the model

loaded_model = load_model("asl_model.h5")
# load weights into new model

print("Loaded model from disk")

cap = cv2.VideoCapture(0)

# Category dictionary
categories = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',26:'Del',27:'Nothing',28:'Space'}

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
    frame = cv2.flip(frame, 1)
    
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
   # _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    #cv2.imshow("test", test_image)
    # Batch of 1
    test_image = roi
    result = loaded_model.predict(test_image.reshape(1, 200, 200, 3))
    prediction = {'A': result[0][0], 
                  'B': result[0][1], 
                  'C': result[0][2],
                  'D': result[0][3],
                  'E': result[0][4],
                  'F': result[0][5],
                  'G': result[0][6],
                  'H': result[0][7],
                   'I': result[0][8],
                    'J': result[0][9],
                     'K': result[0][10],
                      'L': result[0][11],
                       'M': result[0][12],
                        'N': result[0][13],
                         'O': result[0][14],
                          'P': result[0][15],
                           'Q': result[0][16],
                            'R': result[0][17],
                             'S': result[0][18],
                     'T': result[0][19],
                      'U': result[0][20],
                       'V': result[0][21],
                        'W': result[0][22],
                         'X': result[0][23],
                          'Y': result[0][24],
                           'Z': result[0][25],
                            'Del': result[0][26],
                             'Nothing': result[0][27],
                              'Space': result[0][28],
               }
              
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
    # Displaying the predictions

    cv2.putText(frame,'Output:'+  prediction[0][0],(350, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,300), 2)    
    
    cv2.imshow("Frame", frame)
    
    interrupt = cv2.waitKey(50)
    if interrupt & 0xFF == 27: # esc key
        break
        
 
cap.release()
cv2.destroyAllWindows()
