import torch
import pickle
from torch.utils import data
from Simple-Fusion.HateMemesFusion import Dataset_ViT, collate_fn
from Simple-Fusion.HateMemesFusion import Text_Model, Image_Model, Combined_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FOLDER_NAME = '/backup/girish_datasets/HateMM/'

from datasets import load_dataset
dataset = load_dataset('limjiayi/hateful_memes_expanded')

with open(FOLDER_NAME + 'all_hatememes_ext_test_clip_proj_embedding.pkl', 'rb') as fp:
    TextEmbedding_test = pickle.load(fp)

with open(FOLDER_NAME + 'hatememes_ext_test_CLIP_proj_embedding.pkl', 'rb') as fp:
    ImgEmbedding_test = pickle.load(fp)

ext_data = {}

# DataLoaders
for split in dataset.keys():
    # consider only the first 8.5k samples for training, 500 for validation, and 1k for testing (hateful memes dataset)
    if split == 'train':
        dataset[split] = dataset[split].select(list(range(8500)))
    elif split == 'validation':
        dataset[split] = dataset[split].select(list(range(500)))
    elif split == 'test':
        dataset[split] = dataset[split].select(list(range(1000)))

    ext_data[split] = Dataset_ViT(dataset, split)

input_text_size = 512   # 512 for CLIP, 768 for BERT and HXP
input_image_size = 512  # 768 for CLIP and VIT, 384 for DINOv2
fc1_hidden = 128
fc2_hidden = 128

num_classes = 2
batch_size = 32

test_loader = data.DataLoader(ext_data['test'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

text_model = Text_Model(input_text_size, fc1_hidden, fc2_hidden, num_classes).to(device)
image_model = Image_Model(input_image_size, fc1_hidden, fc2_hidden, num_classes).to(device)

model = Combined_model(text_model, image_model, num_classes).to(device)

# Randomly sample 5 test images, their predictions, and the true labels
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
with torch.no_grad():
    for i in range(5):
        text, image, label = ext_data['test'][i]
        text = text.unsqueeze(0).to(device)
        image = image.unsqueeze(0).to(device)

        output = model(text, image)
        _, predicted = torch.max(output.data, 1)
        print(f"Predicted: {predicted.item()}, True label: {label}")
