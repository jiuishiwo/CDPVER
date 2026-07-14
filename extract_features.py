# save_features.py
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../../'))
import numpy as np
from emotiefflib.facial_analysis import EmotiEffLibRecognizer
import cv2
from tqdm import tqdm



import torch
torch.set_num_threads(4)

DATASET_ROOT = r'F:\Computer\Works\Postgraduate\EmotiEffLib-main\dataset\AFEW'
TRAIN_FACES_DIR = os.path.join(DATASET_ROOT, 'Train','AlignedFaces_LBPTOP_Points','Faces')
VAL_FACES_DIR = os.path.join(DATASET_ROOT, 'Val','AlignedFaces_LBPTOP_Points_Val','Faces')
TRAIN_FEATURE_SAVE_DIR = os.path.join(DATASET_ROOT, 'Train', 'video_features_original_b2_7')
VAL_FEATURE_SAVE_DIR = os.path.join(DATASET_ROOT, 'Val', 'video_features_original_b2_7')
MODEL_NAME = 'enet_b2_7'
DEVICE = 'cpu'

def load_face_images(face_dir):
    images = []
    for img_name in sorted(os.listdir(face_dir)):
        img_path = os.path.join(face_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return images

def extract_for_split(faces_dir, feature_save_dir, recognizer, split_name, batch_size=8):
    os.makedirs(feature_save_dir, exist_ok=True)
    # 直接遍历所有数字编号文件夹
    video_ids = [d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))]
    for video_id in tqdm(video_ids, desc=f'Extracting features ({split_name})'):
        video_dir = os.path.join(faces_dir, video_id)
        save_path = os.path.join(feature_save_dir, f"{video_id}.npy")
        if os.path.exists(save_path):
            continue  # 已经提取过的跳过
        face_imgs = load_face_images(video_dir)
        if len(face_imgs) == 0:
            continue
        features_list = []
        for i in range(0, len(face_imgs), batch_size):
            batch_imgs = face_imgs[i:i+batch_size]
            batch_features = recognizer.extract_features(batch_imgs)
            features_list.append(batch_features)
        features = np.concatenate(features_list, axis=0)
        np.save(save_path, features)
        tqdm.write(f"Saved features for video {video_id}, shape: {features.shape}")

def main():
    recognizer = EmotiEffLibRecognizer(engine='torch', model_name=MODEL_NAME, device=DEVICE)
    extract_for_split(TRAIN_FACES_DIR, TRAIN_FEATURE_SAVE_DIR, recognizer, 'train', batch_size=16)
    extract_for_split(VAL_FACES_DIR, VAL_FEATURE_SAVE_DIR, recognizer, 'val', batch_size=16)

if __name__ == '__main__':
    main()