from PIL import Image
import requests, os
import numpy as np
import pickle
from tqdm import tqdm

from transformers import AutoImageProcessor, AutoModel

FOLDER_NAME = '/backup/girish_datasets/HateMM/'
DINOv2_FOLDER = os.path.join(FOLDER_NAME, 'DINOv2_lhs/')

if not os.path.exists(DINOv2_FOLDER):
    os.makedirs(DINOv2_FOLDER)
    print(f"Created directory: {DINOv2_FOLDER}")
else:
    print(f"Directory already exists: {DINOv2_FOLDER}")

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model = AutoModel.from_pretrained('facebook/dinov2-small')
model.to("cuda")

minFrames = 100

with open(FOLDER_NAME+'final_allImageFrames.pkl', 'rb') as fp:
    allDataAnnotation = pickle.load(fp)
    allVidList = list(allDataAnnotation.values())

def read_images(frame_paths):
    X = []
    currFrameCount = 0
    videoFrameCount = len(frame_paths)
    if videoFrameCount <= minFrames:
        for frame_path in frame_paths:
            image = Image.open(frame_path)    

            X.append(image)
            currFrameCount += 1
            if(currFrameCount==minFrames):
                break
        paddingImage = Image.fromarray(np.zeros((100,100)), 'RGB')
        while currFrameCount < minFrames:
            X.append(paddingImage)
            currFrameCount+=1
    else:
        step = int(videoFrameCount/minFrames)
        for i in range(0,videoFrameCount,step):
            image = Image.open(frame_paths[i])
            X.append(image)
            currFrameCount += 1
            if(currFrameCount==minFrames):
                break
        paddingImage = Image.fromarray(np.zeros((100,100)), 'RGB')
        while currFrameCount < minFrames:
            X.append(paddingImage)
            currFrameCount+=1
    return X 


for vid in tqdm(allVidList):
    video_id = os.path.basename(os.path.dirname(vid[0]))
    pickle_filename = os.path.join(DINOv2_FOLDER, f'{video_id}_DINOv2_features.pkl')
    if os.path.exists(pickle_filename):
        continue
    try:
        images = read_images(vid)
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        video_features = [(last_hidden_states[i][0].cpu().detach().numpy()) for i in range(0,100)]
        # pooled_output = outputs.pooler_output  # pooled CLS states
        with open(pickle_filename, 'wb') as fp:
            pickle.dump(video_features, fp)
    except Exception as e:
        print(f"Error occurred while processing {video_id}: {e}")
        continue
