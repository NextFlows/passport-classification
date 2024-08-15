import argparse
import os
import os.path as osp

import cv2
import tensorflow as tf

from tqdm import tqdm
from model.classify import PassportClassifier
from utils.compute_metrics import compute_metrics
from utils.image_rotation import rotate_image_randomly


def limit_tf_memory():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPU')
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

def save_img(save_dir, img_name, img):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(os.path.join(save_dir,img_name), img)

def test(base_path, classifier, save_wrong_folder=None):
    labels, preds = [], []
    for label in ['passport', 'non-passport']:
        # Folder name
        folder = os.path.join(base_path, label)
        if not os.path.isdir(folder):
            continue

        # Run model on all images
        for file in tqdm(list(os.listdir(folder))):
            image = cv2.imread(os.path.join(folder,file))
            try:
                pred = classifier.classify(image)
                preds.append(pred)
                labels.append(label)
            except:
                print(f'Error on {label}/{file}')
                continue

            # Save wrong preds
            if pred != label:
                os.makedirs(save_wrong_folder, exist_ok=True)
                save_name = os.path.join(save_wrong_folder,
                                         f'l_{label}_p_{pred}_{file}')
                cv2.imwrite(save_name, image)

    # Compute metrics
    compute_metrics(labels, preds)

def main(config):
    limit_tf_memory()
    classifier = PassportClassifier(pth_device='cuda:0')

    if config.mode == 'test':
        test(config.images_path,
             classifier,
             save_wrong_folder=config.save_wrong_folder)

    else:
        if os.path.isfile(config.images_path):
            img_paths = [config.images_path]
        else:
            img_paths = [os.path.join(config.images_path, f)
                         for f in os.listdir(config.images_path)]

        for img_path in img_paths:
            img = cv2.imread(img_path)
            try:
                pred = classifier.classify(img)
                print(f'{img_path} ---> {pred}')
            except:
                print(f'Error on {img_path}')
                continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_path', type=str,
                        help='path of input image or directory of images')
    parser.add_argument('--save_dir', type=str, help='output path',
                        default='result')
    parser.add_argument('--mode', type=str, help='mode to switch between test and predict',
                        default='predict')
    parser.add_argument('--save_wrong_folder', type=str, help='folder to save images with wrong prediction in',
                        default='results/errors')

    args = parser.parse_args()
    main(args)
