import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
import numpy as np
import math


def biggest_contour(contor):
        biggest=np.array([])

        max_area=0
        for i in contor:
            area=cv2.contourArea(i)
            if area>=1000:
                peri=cv2.arcLength(i,True)
                approx=cv2.approxPolyDP(i,0.02*peri,True)
                if area>max_area and len(approx)==4:
                    biggest=approx
                    max_area=area
        return biggest
webcam=cv2.VideoCapture(1)
def order_contour(pts):
     if pts.size != 0:
         pts = pts.reshape(4, 2)
         s = pts.sum(axis=1)
         d = np.diff(pts, axis=1)
         tl = pts[np.argmin(s)]
         br = pts[np.argmax(s)]
         tr = pts[np.argmin(d)]
         bl = pts[np.argmax(d)]
         return np.array([tl, tr, br, bl], dtype=np.int32)
def distance(p1,p2):
     return math.dist(p1,p2)
while True:
    is_true,frames=webcam.read()
  
    if not is_true:
        break
    
    fps = webcam.get(cv2.CAP_PROP_FPS)

    cv2.putText(frames,f'FPS: {int(fps)}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,156))
    
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    gray=cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)
    clahe_img=clahe.apply(gray)
    blurrimg=cv2.bilateralFilter(gray,9,75,75)
    kernel=np.ones((3,3),np.uint8)
    edge=cv2.Canny(blurrimg,50,100)
    erode=cv2.erode(edge,kernel,iterations=1)
    dilate=cv2.dilate(erode,kernel,iterations=1)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contor=sorted(contours,key=cv2.contourArea,reverse=True)[:10]
 
    big_contor = biggest_contour(contor)
    if big_contor.size != 0:
        ordered = order_contour(big_contor)
        tl, tr, br, bl = ordered
        scale=10
        w_real, h_real = 22, 28.23
        
        dst_pts = np.array([[0, 0], [w_real*scale, 0], [w_real*scale, h_real*scale], [0, h_real*scale]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(ordered.astype(np.float32), dst_pts)
        warped = cv2.warpPerspective(frames, M, (int(w_real*scale), int(h_real*scale)))
        #width_px = distance(tr, br)
        #height_px = distance(tl, tr)
        px_per_unit_w = warped.shape[1]/ w_real if w_real != 0 else 0
        px_per_unit_h = warped.shape[0]/ h_real if h_real != 0 else 0
        pixel_per_cm = (px_per_unit_w + px_per_unit_h) / 2
        warped_gray= cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped_edge= cv2.Canny(warped_gray, 50, 100)
        contours_warped, _ = cv2.findContours(warped_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        for c in  contours_warped:
            if cv2.contourArea(c) < 500:
                continue
            #rectangle = cv2.minAreaRect(c)
            x,y,w,h=cv2.boundingRect(c)
            #(cx, cy), (w_px, h_px), angle = rectangle
            w_obj = w/ px_per_unit_w if px_per_unit_w != 0 else 0
            h_obj = h/ px_per_unit_h if px_per_unit_h != 0 else 0
            cv2.rectangle(warped, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(warped, f"{w_obj:.2f}cm x {h_obj:.2f}cm", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.imshow("Measured Objects (Warped)", warped)
        cv2.drawContours(frames, [big_contor], -1, (0, 255, 0), 3)
        cv2.imshow("frames", frames)
        
    else:
        cv2.imshow("frames", frames)

    if cv2.waitKey(40) & 0xFF ==ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()
