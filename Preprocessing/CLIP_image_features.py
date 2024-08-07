from PIL import Image
import requests, os
import numpy as np
import pickle
from tqdm import tqdm

from transformers import AutoProcessor, CLIPVisionModel

FOLDER_NAME = '/backup/girish_datasets/HateMM/'
CLIP_FOLDER = os.path.join(FOLDER_NAME, 'CLIP_lhs/')

if not os.path.exists(CLIP_FOLDER):
    os.makedirs(CLIP_FOLDER)
    print(f"Created directory: {CLIP_FOLDER}")
else:
    print(f"Directory already exists: {CLIP_FOLDER}")

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
model.to("cuda")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
    pickle_filename = os.path.join(CLIP_FOLDER, f"{video_id}_clip.pkl")
    if os.path.exists(pickle_filename):
        continue
    try:
        images = read_images(vid)
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        # pooled_output = outputs.pooler_output  # pooled CLS states
        video_features = [(last_hidden_state[i][0].cpu().detach().numpy()) for i in range(0,100)]
        with open(pickle_filename, 'wb') as f:
            pickle.dump(video_features, f)
    except Exception as e:
        print(f"Error occurred while processing {video_id}: {e}")
        continue
