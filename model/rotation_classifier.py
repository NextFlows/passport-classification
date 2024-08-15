from retinaface import RetinaFace
import cv2
import numpy as np
import math
import os
from utils.image_rotation import correct_image_rotation

def calculate_distance(points, angle):
    left_eye = points[1]
    nose = points[2]
    
    if angle >= 0 and angle<=40: 
        dist_of_nose = math.sqrt((nose[0]-nose[0])**2 + (nose[1]-0)**2)
        dist_of_eye = math.sqrt((left_eye[0]-left_eye[0])**2 + (left_eye[1]-0)**2)
        if dist_of_eye > dist_of_nose:
            return 180
        return 0    
    
    elif angle >= 60 and angle<=90:
        dist_of_nose = math.sqrt(nose[0]**2 + (nose[1]-nose[1])**2)
        dist_of_eye = math.sqrt(left_eye[0]**2 + (left_eye[1]-left_eye[1])**2)
        if dist_of_eye > dist_of_nose:
            return 270
        return 90


class RotationClassifier():
    def __init__(self) -> None:
        self.count=0
   
    def calculate_rotation(self, resp):
    
        points=[]
        if len(resp)>1:
            
            max_conf_face = resp['face_1'] 
            for face in resp.values():
                if face['score'] > max_conf_face['score']:
                    max_conf_face = face
            resp = max_conf_face
        elif len(resp) == 1:
            resp = resp['face_1']
            
        if resp:
            points.append(resp['landmarks']['right_eye'])
            points.append(resp['landmarks']['left_eye'])
            points.append(resp['landmarks']['nose'])
        
        x1 = int(resp['landmarks']['right_eye'][0])
        y1 = int(resp['landmarks']['right_eye'][1])
        x2 = int(resp['landmarks']['left_eye'][0])
        y2 = int(resp['landmarks']['left_eye'][1])
        
        if x2 != x1: 
            m1 = (y2 - y1)/ (x2 - x1) 
        else:
            m1 = (y2 - y1)/ (1)
            
        angle = abs(math.atan(m1)*(180/math.pi))
        rot = calculate_distance(points, angle)
        
        if rot in range(0, 360):
            return rot
        return 0 

    def predict(self, img): 
        resp = RetinaFace.detect_faces(img)
        if isinstance(resp, dict):
            return self.calculate_rotation(resp)
            
        else:
            img = correct_image_rotation(img, 180)
            resp = RetinaFace.detect_faces(img)
            if isinstance(resp, dict):
                return 180 + self.calculate_rotation(resp)

        return 0
