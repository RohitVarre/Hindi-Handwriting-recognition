#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
from keras.models import load_model
import numpy as np
from collections import deque
cv2. __version__ 


# In[ ]:


model1=load_model('devanagiri.h5')
print(model1)


# In[ ]:


letter_count={0:'ka', 1:'kha', 2:'ga', 3:'gha', 4:'kna', 5:'cha', 6:'chha', 7:'ja', 8:'jha', 9:'yna', 
             10:'tamaatar', 11:'thaa', 12:'daa', 13:'dhaa', 14:'adna', 15:'tabala', 16:'tha', 17:'da',
             18:'dha', 19:'na', 20:'pa', 21:'pha', 22:'ba', 23:'bha', 24:'ma', 25:'yaw', 26:'ra', 27:'la',
             28:'waw', 29:'motosaw', 30:'petchiryakha', 31:'patalosaw', 32:'ha', 33:'chhya', 34:'tra', 35:'gya', 36:'0',
             37:'1', 38:'2', 39:'3', 40:'4', 41:'5', 42:'6', 43:'7', 44:'8', 45:'9'}


# In[ ]:


def keras_predict(model,image):
    processed=keras_process_image(image)
    print("processed"+str(processed.shape))
    pred_probab=model.predict(processed)[0]
    pred_class=list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def keras_process_image(imag):
    image_x=32
    image_y=32
    img=cv2.resize(imag,(image_x,image_y))
    img=np.array(img,dtype=np.float32)
    img=np.reshape(img,(-1,image_x,image_y,1))
    return img


# In[ ]:


cap=cv2.VideoCapture(0)
Lower_blue=np.array([110,50,50])
Upper_blue=np.array([130,255,255])
pred_class=0
pts=deque(maxlen=512)
blackboard=np.zeros((480,640,3), dtype=np.uint8)
digit=np.zeros((200,200,3), dtype=np.uint8)
while(cap.isOpened()):
    ret,img=cap.read()
    img=cv2.flip(img,1)
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(imgHSV,Lower_blue,Upper_blue)
    blur=cv2.medianBlur(mask,15)
    blur=cv2.GaussianBlur(blur,(5,5),0)
    thresh=cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cnts=cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    center=None
    if len(cnts)>=1:
        contour=max(cnts, key=cv2.contourArea)
        if cv2.contourArea(contour)>250:
            (x,y),radius=cv2.minEnclosingCircle(contour)
            cv2.circle(img, (int(x), int(y)), int(radius), (0,255,255), 2)
            cv2.circle(img, center, 5, (0,0,255), -1)
            M=cv2.moments(contour)
            center=int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            pts.appendleft(center)
            for i in range(1,len(pts)):
                if pts[i-1] is None or pts[i] is None:
                    continue
                cv2.line(blackboard, pts[i-1], pts[i], (255,255,255), 10)
                cv2.line(img, pts[i-1], pts[i], (0,0,255), 5)
    elif len(cnts)==0:
        if len(pts)!=[]:
            blackboard_grey=cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
            blur1=cv2.medianBlur(blackboard_grey, 15)
            blur1=cv2.GaussianBlur(blur1,(5,5), 0)
            thresh1=cv2.threshold(blur1,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            blackboard_cnts=cv2.findContours(thresh1.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1]
            if len(blackboard_cnts)>=1:
                cnts=max(blackboard_cnts,key=cv2.contourArea)
                print(cv2.contourArea(cnts))
                if cv2.contourArea(cnts) >= 2000:
                    x,y,w,h=cv2.boundingRect(cnts)
                    digit=blackboard_grey[y:y+h,x:x+w]
                    pred_probab,pred_class=keras_predict(model1,digit)
                    print(pred_class,pred_probab)
                        
        pts=deque(maxlen=512)
        blackboard=np.zeros((480,640,3), dtype=np.uint8)
    cv2.putText(img, "Conv Network= " + str(letter_count[pred_class]), (10,470), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)
    cv2.imshow("Frame",img)
    cv2.imshow("Contours",thresh)
    k=cv2.waitKey(10)
    if k==27:
        break
                        


# In[ ]:





# In[ ]:




