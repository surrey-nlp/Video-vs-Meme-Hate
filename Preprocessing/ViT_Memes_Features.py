import torch
from tqdm import tqdm
import os, requests
import pickle
from PIL import Image
from io import BytesIO

from transformers import ViTFeatureExtractor, ViTModel

from datasets import load_dataset
dataset = load_dataset('limjiayi/hateful_memes_expanded')

FOLDER_NAME = '/backup/girish_datasets/HateMM/'

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

def extract_image_features(image_id_hatexplain, processor, model, device):
    try:
        # Check if the image ID ends with '.png' or '.jpg'
        if not image_id_hatexplain.endswith('.png') and not image_id_hatexplain.endswith('.jpg'):
            image_id_hatexplain += '.png'
        
        # Check if the image has already been processed
        if image_id_hatexplain in processed_hxp_ids:
            print(f"Skipping image with ID: {image_id_hatexplain}")
            return None

        # Download the image from the Hugging Face repository
        image_url = f"https://huggingface.co/datasets/limjiayi/hateful_memes_expanded/resolve/main/img/{image_id_hatexplain}"
        response = requests.get(image_url)

        if response.status_code == 200:
            image_bytes = response.content
            image = Image.open(BytesIO(image_bytes))
            inputs = processor(images=image, return_tensors="pt")

            # Ensure input tensor is on the same device as model's weights
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                image_embeds = outputs.image_embeds
                return image_embeds[0].cpu().numpy()
        else:
            print(f"Error downloading image: {image_id_hatexplain}")
            return None
    except Exception as e:
        print(f"Error processing image: {image_id_hatexplain}")
        print(e)
        return None

def save_features_to_pickle(features, split):
    pickle_file = FOLDER_NAME + f'hatememes_ext_{split}_ViT_embedding.pkl'
    if os.path.exists(pickle_file) and os.path.getsize(pickle_file) > 0:
        with open(pickle_file, 'rb') as fp:
            existing_data = pickle.load(fp)
            existing_data.update(features)
    else:
        existing_data = features

    with open(pickle_file, 'wb') as fp:
        pickle.dump(existing_data, fp)

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
print(device)

# Move the model to the GPU
model.to(device)

# Load processed IDs from a file
processed_hxp_ids = set()
try:
    with open('processed_img_test_ids.txt', 'r') as file:
        for line in file:
            processed_hxp_ids.add(line.strip())
except FileNotFoundError:
    pass

print("Starting image processing for ViT...")
split = 'train'
ImgEmbedding_train = {}
print(f"Processing split: {split}")

for example in tqdm(dataset[split]):
    image_id_hatexplain = example['id']
    image_embeds = extract_image_features(image_id_hatexplain, feature_extractor, model, device)
    if image_embeds is not None:
        ImgEmbedding_train[image_id_hatexplain] = image_embeds

save_features_to_pickle(ImgEmbedding_train, split)

print("Image processing for hateful memes completed.")
