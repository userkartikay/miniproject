import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
import numpy as np

'''model=YOLO("yolo11m.pt")
model.train(data="C:\\Users\\karti\\OneDrive\\Desktop\\mini project\\pencil.v1i.yolov8\\data.yaml",imgsz=640,batch=8,epochs=100)
'''
webcam=cv2.VideoCapture(1)
while True:
    is_true,frames=webcam.read()
  
    if not is_true:
        break
    
    fps = webcam.get(cv2.CAP_PROP_FPS)

    cv2.putText(frames,f'FPS: {int(fps)}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,156))
    
    
    gray=cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)
    blurrimg=cv2.GaussianBlur(gray,(3,3),0)
    edge=cv2.Canny(blurrimg,50,100)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contor=sorted(contours,key=cv2.contourArea,reverse=True)[:10]
    def biggest_contour(contor):
        biggest=np.array([])

        max_area=0
        for i in contours:
            area=cv2.contourArea(i)
            if area>=1000:
                peri=cv2.arcLength(i,True)
                approx=cv2.approxPolyDP(i,0.015*peri,True)
                if area>max_area and len(approx)==4:
                    biggest=approx
                    max_area=area
        return biggest
    big_contor=biggest_contour(contor)
    cv2.drawContours(frames,[big_contor],-1,(0,255,0),3)

    


    cv2.imshow("frames",frames)

    if cv2.waitKey(40) & 0xFF ==ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()
