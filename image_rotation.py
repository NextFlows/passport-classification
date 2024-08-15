import argparse
import os
import cv2

from model.rotation_classifier import RotationClassifier 
from utils.image_rotation import correct_image_rotation


def main(config):
    
    classifier = RotationClassifier()

    if os.path.isfile(config.images_path):
        img_paths = [config.images_path]
    else:
        img_paths = [os.path.join(config.images_path, f)
                        for f in os.listdir(config.images_path)]
    
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
        
    for img_path in img_paths:
        img = cv2.imread(img_path)
        angle = classifier.predict(img)  
        print(f'{img_path} ---> ', angle)
        img = correct_image_rotation(img, angle)      
        output_path = os.path.join(config.save_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, img)    
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_path', type=str,
                        help='path of input image or directory of images')
    parser.add_argument('--save_dir', type=str, help='output path',
                        default='output')
    args = parser.parse_args()
    main(args)
