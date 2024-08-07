import os
import pickle
from datasets import load_dataset
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

FOLDER_NAME = '/backup/girish_datasets/HateMM/'

# from transformers import AutoTokenizer, ClapTextModel

# model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
# model = ClapTextModel.from_pretrained("laion/clap-htsat-unfused")
# tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

model.to('cuda')

dataset = load_dataset('limjiayi/hateful_memes_expanded')

def process_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        text_embeds = outputs.text_embeds[0].cpu().detach().numpy()
    return text_embeds

def save_embeddings(embeddings, pickle_file):
    if os.path.exists(pickle_file) and os.path.getsize(pickle_file) > 0:
        with open(pickle_file, 'rb') as fp:
            existing_data = pickle.load(fp)
    else:
        existing_data = {}
    existing_data.update(embeddings)
    with open(pickle_file, 'wb') as fp:
        pickle.dump(existing_data, fp)

def update_processed_ids(text_id, processed_ids_file):
    with open(processed_ids_file, 'a') as file:
        file.write(text_id + '\n')

def update_skipped_samples(text_id, skipped_samples_file):
    with open(skipped_samples_file, 'a') as file:
        file.write(f"{text_id}\n")

processed_ids_file = 'processed_text_ids.txt'
skipped_samples_file = 'skipped_samples.txt'

print("Starting processing for CLIP text...")
split = 'train'
# for split in ['train', 'validation', 'test']:
print(f"Processing split: {split}")
allEmbedding_hatexplain = {}
skipped_samples = set()  # Define the skipped_samples variable
for example in tqdm(dataset[split]):
    try:
        text_id_hatexplain = example['id']
        if text_id_hatexplain in skipped_samples:
            print(f"Skipping text with ID: {text_id_hatexplain}")
            continue
        processed_hxp_ids = set()  # Define the processed_hxp_ids variable
        if text_id_hatexplain in processed_hxp_ids:
            print(f"Skipping text with ID: {text_id_hatexplain}")
            continue

        text = example['text']
        embeddings = process_text(text, model, tokenizer)
        allEmbedding_hatexplain[text_id_hatexplain] = embeddings

        pickle_file = FOLDER_NAME + f'all_hatememes_ext_{split}_clip_proj_embedding.pkl'
        save_embeddings(allEmbedding_hatexplain, pickle_file)

        if text_id_hatexplain not in processed_hxp_ids:
            update_processed_ids(text_id_hatexplain, processed_ids_file)

    except Exception as e:
        print(f"Error processing text with ID: {text_id_hatexplain}. Skipping this sample.")
        update_skipped_samples(text_id_hatexplain, skipped_samples_file)
        continue

print(f"Finished processing split: {split}")