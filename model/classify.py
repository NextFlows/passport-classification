import os.path as osp
import re
import gdown

import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from retinaface import RetinaFace
from transformers import ViTFeatureExtractor, pipeline

from model.lit_model import LitClassifier
from model.rotation_classifier import RotationClassifier
from utils.image_rotation import correct_image_rotation
import os

class PassportClassifier:
    def __init__(self, pth_device='cpu', model_path='weights', gdrive_id=None):
        if gdrive_id is not None:
            self.maybe_download_weights(model_path, gdrive_id)

        self.device = pth_device
        self.clf = LitClassifier(num_labels=4,
                                 model_path=osp.join(model_path, "vit")).to(pth_device)
        ckpt_path = osp.join(model_path, "idc_vit.pth")
        self.clf.load_state_dict(torch.load(ckpt_path, map_location=pth_device))
        self.clf.eval()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            osp.join(model_path, "vit")
        )
        self.ocr = easyocr.Reader(['ch_sim','en'], gpu=False)
        self.classifier = RotationClassifier()

    def maybe_download_weights(self, model_path, gdrive_id):
        if not os.path.exists(os.path.join(model_path, 'vit')):
            os.makedirs(model_path, exist_ok=True)
            gdown.download_folder(gdrive_id, quiet=False)
        else:
            print(f'Folder {model_path} already exists, so skipping downloading weights')
            print(f'Note that the model may not run successfully if weights are corrupted')


    def crop_margins(self, img):
        extracted_text = self.ocr.readtext(img)
        if len(extracted_text) > 0:
            xs = [int(cc[0]) for c in extracted_text for cc in c[0]]
            ys = [int(cc[1]) for c in extracted_text for cc in c[0]]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            img = img[min_y:max_y, min_x:max_x]
        return img

    @torch.no_grad()
    def classify(self, img):
        """Image should ibn BGR format"""

        rotation = self.classifier.predict(img)
        img = correct_image_rotation(img, rotation)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.crop_margins(img)

        # Check if face is present in image
        # All passports have a face image in them
        out = RetinaFace.detect_faces(img, threshold=0.5)
        is_face_detected = isinstance(out, dict)
        if not is_face_detected:
            return "This is not a passport"

        # Extract features using pre-trained transformer
        img_processed = self.feature_extractor(images=img,
                                               return_tensors="pt")
        img_processed = img_processed["pixel_values"]
        img_array = img_processed.to(self.device)

        logits = self.clf(img_array).cpu().numpy()
        prediction = np.argsort(logits)
        prediction = prediction[0][2:]

        # if face detected and model predicted passport, return passport
        if 3 in prediction:
            return "This is a passport"

        # if face detected and special text combination appears in text
        extracted_text = self.ocr.readtext(img)
        text = ""
        for word in extracted_text:
            text += word[1] + " "
        new_string = text.replace("〈", "<" )
        #is_match = re.findall(r"\b[p|P]<?[a-zA-Z]*[<*| ][a-zA-Z0-9]*<+", new_string)
        is_match = '<<' in new_string
        if is_match:
            return "This is a passport"

        return "This is not a passport"
