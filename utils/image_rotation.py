import cv2
import numpy as np


rotations_key = {0:None, 90:cv2.ROTATE_90_COUNTERCLOCKWISE, 
             180:cv2.ROTATE_180, 270:cv2.ROTATE_90_CLOCKWISE}
rotations = [0, 90, 180, 270]

def correct_image_rotation(image, rotation):
    
    if not rotation == 0:
        image = cv2.rotate(image, rotations_key.get(360 - rotation))
    return image

def rotate_image_randomly(image):
    random_rotation = np.random.choice(rotations, p=[0.5, 0.17, 0.17, 0.16])
    if not random_rotation == 0:
        image = cv2.rotate(image, rotations_key.get(random_rotation))
    return random_rotation, image 
