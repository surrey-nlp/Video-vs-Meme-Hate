import pickle
import torch
import numpy as np
from torch import nn
from torch.utils import data
from transformers import BartTokenizerFast
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from multimodal_bart_downstream import MultimodalBartForSequenceClassification
from transformers import BartModel
from audio_video_first import MultimodalAudio

from datasets import load_dataset
dataset = load_dataset('limjiayi/hateful_memes_expanded')

FOLDER_NAME = '/backup/girish_datasets/HateMM/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-4

SOURCE_MAX_LEN = 768 # 500
ACOUSTIC_DIM = 768
ACOUSTIC_MAX_LEN = 1000
VISUAL_DIM = 384 # 2048
VISUAL_MAX_LEN = 100 # 480


# with open(FOLDER_NAME + 'hatememes_ext_train_VITembedding.pkl', 'rb') as fp:
with open(FOLDER_NAME + 'hatememes_ext_train_DINOv2embedding.pkl', 'rb') as fp:
# with open(FOLDER_NAME + 'hatememes_ext_train_CLIPembedding.pkl', 'rb') as fp:
    ImgEmbedding_train = pickle.load(fp)

# with open(FOLDER_NAME + 'all_hatememesext_train_rawBERTembedding.pkl', 'rb') as fp:
# # with open(FOLDER_NAME + 'all_hatememesext_train_hatexplain_embedding.pkl', 'rb') as fp:
#     TextEmbedding_train = pickle.load(fp)

# with open(FOLDER_NAME + 'all_hatememesext_validation_rawBERTembedding.pkl', 'rb') as fp:
# # with open(FOLDER_NAME + 'all_hatememesext_validation_hatexplain_embedding.pkl', 'rb') as fp:
#     TextEmbedding_val = pickle.load(fp)

# with open(FOLDER_NAME + 'hatememes_ext_validation_VITembedding.pkl', 'rb') as fp:
# with open(FOLDER_NAME + 'hatememes_ext_validation_CLIPembedding.pkl', 'rb') as fp:
with open(FOLDER_NAME + 'hatememes_ext_validation_DINOv2embedding.pkl', 'rb') as fp:
    ImgEmbedding_val = pickle.load(fp)

# with open(FOLDER_NAME + 'all_hatememesext_test_rawBERTembedding.pkl', 'rb') as fp:
# # with open(FOLDER_NAME + 'all_hatememesext_test_hatexplain_embedding.pkl', 'rb') as fp:
#     TextEmbedding_test = pickle.load(fp)

# with open(FOLDER_NAME + 'hatememes_ext_test_VITembedding.pkl', 'rb') as fp:
# with open(FOLDER_NAME + 'hatememes_ext_test_CLIPembedding.pkl', 'rb') as fp:
with open(FOLDER_NAME + 'hatememes_ext_test_DINOv2embedding.pkl', 'rb') as fp:
    ImgEmbedding_test = pickle.load(fp)


model = MultimodalBartForSequenceClassification.from_pretrained("facebook/bart-base")       # for text first model
# bart_model = BartModel.from_pretrained('facebook/bart-base')
tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
# print("Tokenizer : ", tokenizer)

# num_param = sum(p.numel() for p in model.parameters())
# print("Total parameters : ", num_param/1e6)


p = {
        'additional_special_tokens' : ['[CONTEXT]', '[UTTERANCE]']
        # 'additional_special_tokens' : ['[UTTERANCE]']
    }

tokenizer.add_special_tokens(p)

# model = MultimodalAudio()     # for image first model


class Dataset_ViT(data.Dataset):
    def __init__(self, dataset, split='train'):
        "Initialization"
        self.dataset = dataset
        self.split = split
    
    def load_data_for_image(self, image_id):
        try:
            # Find the index of the text corresponding to the image_id
            text_index = self.dataset[self.split]['id'].index(image_id)
        except ValueError:
            print(f"Warning: Invalid image_id {image_id}")
            return None

        # Load text and image data
        try:
            if self.split == 'train':
                text_data = tokenizer(self.dataset[self.split]['text'][text_index], return_tensors='pt', padding='max_length', max_length=SOURCE_MAX_LEN, truncation=True)
                image_data = torch.tensor(np.array(ImgEmbedding_train[self.modify_image_id(image_id)]))
                audio_data = torch.zeros(768, dtype=torch.float32)
            elif self.split == 'validation':
                text_data = tokenizer(self.dataset[self.split]['text'][text_index], return_tensors='pt', padding='max_length', max_length=SOURCE_MAX_LEN, truncation=True)
                image_data = torch.tensor(np.array(ImgEmbedding_val[self.modify_image_id(image_id)]))
                audio_data = torch.zeros(768, dtype=torch.float32)
            else:
                text_data = tokenizer(self.dataset[self.split]['text'][text_index], return_tensors='pt', padding='max_length', max_length=SOURCE_MAX_LEN, truncation=True)
                image_data = torch.tensor(np.array(ImgEmbedding_test[self.modify_image_id(image_id)]))
                audio_data = torch.zeros(768, dtype=torch.float32)
        except KeyError:
            print(f"KeyError: {image_id}")
            # Assign default values for missing data
            text_data = torch.zeros(768, dtype=torch.float32)
            image_data = torch.zeros(768, dtype=torch.float32)
            audio_data = torch.zeros(768, dtype=torch.float32)

        return text_data, image_data, audio_data
    
    def modify_image_id(self, image_id):
        # Append '.png' if not already present
        if not image_id.endswith(('.png', '.jpg')):
            image_id += '.png'
        return image_id

    def __len__(self):
        return len(self.dataset[self.split])

    def __getitem__(self, index):
        "Generates one sample of data"
        image_id = self.dataset[self.split]['id'][index]
        # Load data
        X_text, X_img, X_aud = self.load_data_for_image(image_id)
        # Load label
        y = self.dataset[self.split]['label'][index]

        return X_text['input_ids'], X_text['attention_mask'], X_img, X_aud, y


def collate_fn(batch):
    text_input_ids = [x[0] for x in batch]
    text_attention_mask = [x[1] for x in batch]
    visual_input = [x[2] for x in batch]
    audio_input = [x[3] for x in batch]
    label = [x[4] for x in batch]

    # text_input_ids = torch.stack(text_input_ids)
    # text_attention_mask = torch.stack(text_attention_mask)
    # visual_input = torch.stack(visual_input)
    # audio_input = torch.stack(audio_input)
    # labels = torch.tensor(labels)

    # return text_input_ids, text_attention_mask, visual_input, audio_input, labels

    # Make sure all text tensors have the same shape
    # text_input_ids = [txt.unsqueeze(0) if txt.ndim == 1 else txt for txt in text_input_ids]
    text_input_ids = torch.stack(text_input_ids)
    text_input_ids = text_input_ids.squeeze(1)
    # text_attention_mask = [txt.unsqueeze(0) if txt.ndim == 1 else txt for txt in text_attention_mask]
    text_attention_mask = torch.stack(text_attention_mask)
    text_attention_mask = text_attention_mask.squeeze(1)
    # Make sure all image tensors have the same shape
    image = [img.unsqueeze(0) if img.ndim == 1 else img for img in visual_input]
    image = torch.stack(image)
    # Make sure all audio tensors have the same shape
    # audio = [aud.unsqueeze(0) if aud.ndim == 1 else aud for aud in audio_input]
    audio = torch.stack(audio_input)
    label = torch.tensor(label)
    label = label.unsqueeze(1)

    return text_input_ids, text_attention_mask, image, audio, label
    # batch = list(filter(lambda x: x is not None, batch))
    # if len(batch) == 0:  # Check if the batch is empty after filtering
    #     return None

    # return torch.utils.data.dataloader.default_collate(batch)


# training parameters
k = 2            # number of target category
epochs = 20
batch_size = 32
log_interval = 100

# import wandb
# wandb.init(
#     project="hate-memes-classification",
#     config={
#         "learning_rate": LEARNING_RATE,
#         "architecture": "Multimodal BART + Image@5",
#         "dataset": "Hateful Memes",
#         # "features": "BART + ViT + Wav2Vec2",
#         "epochs": epochs,
#         "batch_size": batch_size,
#     },
# )


ext_data = {}

# DataLoaders
for split in dataset.keys():
    # consider only the first 8.5k samples for training, 500 for validation, and 1k for testing (hateful memes dataset)
    if split == 'train':
        dataset[split] = dataset[split].select(list(range(8500)))
    elif split == 'validation':
        dataset[split] = dataset[split].select(list(range(1000)))
    elif split == 'test':
        dataset[split] = dataset[split].select(list(range(3000)))
    ext_data[split] = Dataset_ViT(dataset, split)

train_loader = data.DataLoader(ext_data['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = data.DataLoader(ext_data['validation'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = data.DataLoader(ext_data['test'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


model.to(DEVICE)

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()


def train_epoch(model, data_loader):
    model.train()
    epoch_train_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_preds = []
    all_labels = []

    for step, batch in enumerate(tqdm(data_loader, desc='Training Iteration')):
        input_ids, attention_mask, visual_input, acoustic_input, labels = batch

        input_ids, attention_mask, visual_input, acoustic_input, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), visual_input.to(DEVICE), acoustic_input.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        acoustic_input=acoustic_input,
                        visual_input=visual_input,
                        labels=labels)

        loss = outputs['loss']
        loss = loss.mean()
        epoch_train_loss += loss.item()

        logits = outputs['logits']
        preds = torch.argmax(logits, dim=-1)
        correct_predictions += (preds == labels).sum().item()
        total_predictions += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        loss.backward()
        optimizer.step()

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        auc_roc = roc_auc_score(all_labels, all_preds)

        # wandb.log({"Train Loss": epoch_train_loss, "Train Accuracy": accuracy, "Train Precision": precision, 
        #            "Train Recall": recall, "Train F1": f1, "Train ROC AUC": auc_roc})
    print("Epoch train loss : ", epoch_train_loss, "Accuracy: ", accuracy, "Precision: ", precision, "Recall: ", recall, "F1 Score: ", f1, "ROC AUC: ", auc_roc)


def valid_epoch(model, data_loader):
    model.eval()
    valid_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc='Validation Iteration')):
            input_ids, attention_mask, visual_input, acoustic_input, labels = batch
            input_ids, attention_mask, visual_input, acoustic_input, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), visual_input.to(DEVICE), acoustic_input.to(DEVICE), labels.to(DEVICE)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            acoustic_input=acoustic_input,
                            visual_input=visual_input,
                            labels=labels)

            logits = outputs['logits']
            loss = outputs['loss']
            loss = loss.mean()

            valid_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='macro')
            recall = recall_score(all_labels, all_preds, average='macro')
            f1 = f1_score(all_labels, all_preds, average='macro')
            auc_roc = roc_auc_score(all_labels, all_preds)

            # wandb.log({"Validation Loss": valid_loss, "Validation Accuracy": accuracy, "Validation Precision": precision, 
            #            "Validation Recall": recall, "Validation F1": f1, "Validation ROC AUC": auc_roc})

    return valid_loss, all_preds, all_labels


def test_epoch(model, data_loader):
    model.eval()
    predictions = []
    gold = []

    correct = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader)):
            input_ids, attention_mask, visual_input, acoustic_input, labels = batch
            input_ids, attention_mask, visual_input, acoustic_input, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), visual_input.to(DEVICE), acoustic_input.to(DEVICE), labels.to(DEVICE)

            outputs = model(input_ids = input_ids,
                            attention_mask = attention_mask,
                            acoustic_input = acoustic_input,
                            visual_input = visual_input,
                            labels = labels)

            logits = outputs['logits']
            pred = logits.argmax(dim = -1)

            predictions.extend(pred.tolist())
            gold.extend(labels.tolist())

            # correct += int((pred == labels).sum())

    return predictions, gold


class EarlyStopping:
  def __init__(self, patience, min_delta):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.min_validation = np.inf

  def early_stop(self, valid_loss):
    if valid_loss < self.min_validation:
      self.min_validation = valid_loss
      self.counter = 0
    elif valid_loss > (self.min_validation + self.min_delta):
      self.counter += 1
      if self.counter >= self.patience:
        return True
    return False
  
early_stopper = EarlyStopping(patience = 15, min_delta = 0.2)


def train_and_validation(model, train_loader, valid_loader):
    best_f1 = 0.0
    for epoch in range(epochs):
      print("\n=============Epoch : ", epoch)
      train_epoch(model, train_loader)
      valid_loss, valid_pred, valid_gold = valid_epoch(model, valid_loader)

      if early_stopper.early_stop(valid_loss):
        break

    #   print("Length of predictions : ", len(valid_pred))
    #   print("Length of gold : ", len(valid_gold))
      print("Valid loss : ", valid_loss)
      print("Valid Accuracy : ", accuracy_score(valid_gold, valid_pred))
      print("Valid Precision : ", precision_score(valid_gold, valid_pred, average = 'macro'))
      print("Valid Recall : ", recall_score(valid_gold, valid_pred, average = 'macro'))
      print("Valid F1 score : ", f1_score(valid_gold, valid_pred, average = 'macro'))
      print("Valid AUC ROC : ", roc_auc_score(valid_gold, valid_pred))

      curr_f1 = f1_score(valid_gold, valid_pred, average = 'macro')

      if(curr_f1 > best_f1):
        best_f1 = curr_f1

        torch.save(model.state_dict(), 'bart_dino_model.pth')
        print("model saved\n")

    return model


# model = train_and_validation(model, train_loader, val_loader)

# test_pred, test_gold = test_epoch(model, test_loader)

# test_accuracy = accuracy_score(test_gold, test_pred)
# test_precision = precision_score(test_gold, test_pred, average = 'macro')
# test_recall = recall_score(test_gold, test_pred, average = 'macro')
# test_f1 = f1_score(test_gold, test_pred, average = 'macro')
# test_auc_roc = roc_auc_score(test_gold, test_pred)

# wandb.log({"Test Accuracy": test_accuracy, "Test Precision": test_precision, "Test Recall": test_recall, 
#            "Test F1": test_f1, "Test ROC AUC": test_auc_roc})

# print("Test accuracy : ", test_accuracy)
# print("Test Precision : ", test_precision)
# print("Test Recall : ", test_recall)
# print("Test F1 score : ", test_f1)
# print("Test AUC ROC : ", test_auc_roc)



def load_model(model_path, model, device):
    # Load the model state dict that was saved after training
    state_dict = torch.load(model_path, map_location=device)

    # If the model was trained using nn.DataParallel, which saves the model with a 'module.' prefix
    # We need to remove this prefix from each key
    new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

    # Load the adjusted state dict into the model
    model.load_state_dict(new_state_dict, strict=False)

    return model

# Test the model for specific memes
test_memes = ['09638', '24098', '43275', '27614', '31208', '47103', '45139', '20634', '78962', '49023', '52894', '72048']

model = load_model('bart_dino_model.pth', model, DEVICE)
model.eval()
test_data = Dataset_ViT(dataset, 'test')
val_data = Dataset_ViT(dataset, 'validation')

with torch.no_grad():
    for meme in test_memes:
        # check if the meme is in the validation set or test set
        if any(meme == val_id for val_id in val_data.dataset['validation']['id']):
            split = 'validation'
            text_data, image_data, audio_data = ext_data[split].load_data_for_image(meme)
        elif any(meme == test_id for test_id in test_data.dataset['test']['id']):
            split = 'test'
            text_data, image_data, audio_data = ext_data[split].load_data_for_image(meme)
        else:
            print(f"Meme ID {meme} not found in the dataset")
            continue

        input_ids = text_data['input_ids'].to(DEVICE)
        attention_mask = text_data['attention_mask'].to(DEVICE)
        visual_input = image_data.to(DEVICE)
        acoustic_input = audio_data.to(DEVICE)

        # print(f"input_ids shape: {input_ids.shape}", f"attention_mask shape: {attention_mask.shape}", 
        #       f"visual_input shape: {visual_input.shape}", f"acoustic_input shape: {acoustic_input.shape}", sep='\n')

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        visual_input=visual_input,
                        acoustic_input=acoustic_input.unsqueeze(0))
        
        logits = outputs['logits']
        preds = torch.argmax(logits, dim=-1)
        
        # print the true label and predicted label
        print(f"Meme ID: {meme}", f"True Label: {dataset[split]['label'][dataset[split]['id'].index(meme)]}", f"Predicted Label: {preds.item()}")

