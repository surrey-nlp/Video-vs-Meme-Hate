import os
from os import listdir
import cv2
from tqdm import tqdm

def extract_frames(video_path, target_folder):
    success, _ = cv2.VideoCapture(video_path).read()
    if not success:
        print(f"Failed to read video: {video_path}")
        return
    
    try:
        os.makedirs(target_folder, exist_ok=True)
    except FileExistsError:
        pass
    
    if os.listdir(target_folder):
        print(f"Frames already extracted for video: {video_path}")
        return
    
    vidcap = cv2.VideoCapture(video_path)
    success = True
    count = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, count * 1000)
        success, img = vidcap.read()
        if not success:
          break
        frame_path = os.path.join(target_folder, f"frame_{count}.jpg")
        cv2.imwrite(frame_path, img)
        count += 1

FOLDER_NAME = './'
target_folder = os.path.join(FOLDER_NAME, 'Dataset_Images')

folder1 = ["Dataset/hate_videos/", "Dataset/non_hate_videos/"]

for subDir in folder1:
    print(subDir)
    for f in tqdm(listdir(os.path.join(FOLDER_NAME, subDir))):
        if f.split('.')[-1] == 'mp4':
          video_path = os.path.join(FOLDER_NAME, subDir, f)
          extract_frames(video_path, os.path.join(target_folder, f.split('.')[0]))